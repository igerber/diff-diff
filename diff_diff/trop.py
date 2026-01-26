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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict

from diff_diff._backend import (
    HAS_RUST_BACKEND,
    _rust_unit_distance_matrix,
    _rust_loocv_grid_search,
    _rust_bootstrap_trop_variance,
    _rust_loocv_grid_search_joint,
    _rust_bootstrap_trop_variance_joint,
)
from diff_diff.results import _get_significance_stars
from diff_diff.utils import compute_confidence_interval, compute_p_value


# Sentinel value for "disabled" mode in LOOCV parameter search
# Following paper's footnote 2: λ=∞ disables the corresponding component
_LAMBDA_INF: float = float('inf')


class _PrecomputedStructures(TypedDict):
    """Type definition for pre-computed structures used across LOOCV iterations.

    These structures are computed once in `_precompute_structures()` and reused
    to avoid redundant computation during LOOCV and final estimation.
    """

    unit_dist_matrix: np.ndarray
    """Pairwise unit distance matrix (n_units x n_units)."""
    time_dist_matrix: np.ndarray
    """Time distance matrix where [t, s] = |t - s| (n_periods x n_periods)."""
    control_mask: np.ndarray
    """Boolean mask for control observations (D == 0)."""
    treated_mask: np.ndarray
    """Boolean mask for treated observations (D == 1)."""
    treated_observations: List[Tuple[int, int]]
    """List of (t, i) tuples for treated observations."""
    control_obs: List[Tuple[int, int]]
    """List of (t, i) tuples for valid control observations."""
    control_unit_idx: np.ndarray
    """Array of never-treated unit indices (for backward compatibility)."""
    D: np.ndarray
    """Treatment indicator matrix (n_periods x n_units) for dynamic control sets."""
    Y: np.ndarray
    """Outcome matrix (n_periods x n_units)."""
    n_units: int
    """Number of units."""
    n_periods: int
    """Number of time periods."""


@dataclass
class TROPResults:
    """
    Results from a Triply Robust Panel (TROP) estimation.

    TROP combines nuclear norm regularized factor estimation with
    exponential distance-based unit weights and time decay weights.

    Attributes
    ----------
    att : float
        Average Treatment effect on the Treated (ATT).
    se : float
        Standard error of the ATT estimate.
    t_stat : float
        T-statistic for the ATT estimate.
    p_value : float
        P-value for the null hypothesis that ATT = 0.
    conf_int : tuple[float, float]
        Confidence interval for the ATT.
    n_obs : int
        Number of observations used in estimation.
    n_treated : int
        Number of treated units.
    n_control : int
        Number of control units.
    n_treated_obs : int
        Number of treated unit-time observations.
    unit_effects : dict
        Estimated unit fixed effects (alpha_i).
    time_effects : dict
        Estimated time fixed effects (beta_t).
    treatment_effects : dict
        Individual treatment effects for each treated (unit, time) pair.
    lambda_time : float
        Selected time weight decay parameter from grid. Note: infinity values
        are converted internally (∞ → 0.0 for uniform weights) for computation.
    lambda_unit : float
        Selected unit weight decay parameter from grid. Note: infinity values
        are converted internally (∞ → 0.0 for uniform weights) for computation.
    lambda_nn : float
        Selected nuclear norm regularization parameter from grid. Note: infinity
        values are converted internally (∞ → 1e10, factor model disabled) for
        computation.
    factor_matrix : np.ndarray
        Estimated low-rank factor matrix L (n_periods x n_units).
    effective_rank : float
        Effective rank of the factor matrix (sum of singular values / max).
    loocv_score : float
        Leave-one-out cross-validation score for selected parameters.
    variance_method : str
        Method used for variance estimation.
    alpha : float
        Significance level for confidence interval.
    n_pre_periods : int
        Number of pre-treatment periods.
    n_post_periods : int
        Number of post-treatment periods (periods with D=1 observations).
    n_bootstrap : int, optional
        Number of bootstrap replications (if bootstrap variance).
    bootstrap_distribution : np.ndarray, optional
        Bootstrap distribution of estimates.
    """

    att: float
    se: float
    t_stat: float
    p_value: float
    conf_int: Tuple[float, float]
    n_obs: int
    n_treated: int
    n_control: int
    n_treated_obs: int
    unit_effects: Dict[Any, float]
    time_effects: Dict[Any, float]
    treatment_effects: Dict[Tuple[Any, Any], float]
    lambda_time: float
    lambda_unit: float
    lambda_nn: float
    factor_matrix: np.ndarray
    effective_rank: float
    loocv_score: float
    variance_method: str
    alpha: float = 0.05
    n_pre_periods: int = 0
    n_post_periods: int = 0
    n_bootstrap: Optional[int] = field(default=None)
    bootstrap_distribution: Optional[np.ndarray] = field(default=None, repr=False)

    def __repr__(self) -> str:
        """Concise string representation."""
        sig = _get_significance_stars(self.p_value)
        return (
            f"TROPResults(ATT={self.att:.4f}{sig}, "
            f"SE={self.se:.4f}, "
            f"eff_rank={self.effective_rank:.1f}, "
            f"p={self.p_value:.4f})"
        )

    def summary(self, alpha: Optional[float] = None) -> str:
        """
        Generate a formatted summary of the estimation results.

        Parameters
        ----------
        alpha : float, optional
            Significance level for confidence intervals. Defaults to the
            alpha used during estimation.

        Returns
        -------
        str
            Formatted summary table.
        """
        alpha = alpha or self.alpha
        conf_level = int((1 - alpha) * 100)

        lines = [
            "=" * 75,
            "Triply Robust Panel (TROP) Estimation Results".center(75),
            "Athey, Imbens, Qu & Viviano (2025)".center(75),
            "=" * 75,
            "",
            f"{'Observations:':<25} {self.n_obs:>10}",
            f"{'Treated units:':<25} {self.n_treated:>10}",
            f"{'Control units:':<25} {self.n_control:>10}",
            f"{'Treated observations:':<25} {self.n_treated_obs:>10}",
            f"{'Pre-treatment periods:':<25} {self.n_pre_periods:>10}",
            f"{'Post-treatment periods:':<25} {self.n_post_periods:>10}",
            "",
            "-" * 75,
            "Tuning Parameters (selected via LOOCV)".center(75),
            "-" * 75,
            f"{'Lambda (time decay):':<25} {self.lambda_time:>10.4f}",
            f"{'Lambda (unit distance):':<25} {self.lambda_unit:>10.4f}",
            f"{'Lambda (nuclear norm):':<25} {self.lambda_nn:>10.4f}",
            f"{'Effective rank:':<25} {self.effective_rank:>10.2f}",
            f"{'LOOCV score:':<25} {self.loocv_score:>10.6f}",
        ]

        # Variance method info
        lines.append(f"{'Variance method:':<25} {self.variance_method:>10}")
        if self.variance_method == "bootstrap" and self.n_bootstrap is not None:
            lines.append(f"{'Bootstrap replications:':<25} {self.n_bootstrap:>10}")

        lines.extend([
            "",
            "-" * 75,
            f"{'Parameter':<15} {'Estimate':>12} {'Std. Err.':>12} "
            f"{'t-stat':>10} {'P>|t|':>10} {'':>5}",
            "-" * 75,
            f"{'ATT':<15} {self.att:>12.4f} {self.se:>12.4f} "
            f"{self.t_stat:>10.3f} {self.p_value:>10.4f} {self.significance_stars:>5}",
            "-" * 75,
            "",
            f"{conf_level}% Confidence Interval: [{self.conf_int[0]:.4f}, {self.conf_int[1]:.4f}]",
        ])

        # Add significance codes
        lines.extend([
            "",
            "Signif. codes: '***' 0.001, '**' 0.01, '*' 0.05, '.' 0.1",
            "=" * 75,
        ])

        return "\n".join(lines)

    def print_summary(self, alpha: Optional[float] = None) -> None:
        """Print the summary to stdout."""
        print(self.summary(alpha))

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert results to a dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing all estimation results.
        """
        return {
            "att": self.att,
            "se": self.se,
            "t_stat": self.t_stat,
            "p_value": self.p_value,
            "conf_int_lower": self.conf_int[0],
            "conf_int_upper": self.conf_int[1],
            "n_obs": self.n_obs,
            "n_treated": self.n_treated,
            "n_control": self.n_control,
            "n_treated_obs": self.n_treated_obs,
            "n_pre_periods": self.n_pre_periods,
            "n_post_periods": self.n_post_periods,
            "lambda_time": self.lambda_time,
            "lambda_unit": self.lambda_unit,
            "lambda_nn": self.lambda_nn,
            "effective_rank": self.effective_rank,
            "loocv_score": self.loocv_score,
            "variance_method": self.variance_method,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert results to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with estimation results.
        """
        return pd.DataFrame([self.to_dict()])

    def get_treatment_effects_df(self) -> pd.DataFrame:
        """
        Get individual treatment effects as a DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with unit, time, and treatment effect columns.
        """
        return pd.DataFrame([
            {"unit": unit, "time": time, "effect": effect}
            for (unit, time), effect in self.treatment_effects.items()
        ])

    def get_unit_effects_df(self) -> pd.DataFrame:
        """
        Get unit fixed effects as a DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with unit and effect columns.
        """
        return pd.DataFrame([
            {"unit": unit, "effect": effect}
            for unit, effect in self.unit_effects.items()
        ])

    def get_time_effects_df(self) -> pd.DataFrame:
        """
        Get time fixed effects as a DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with time and effect columns.
        """
        return pd.DataFrame([
            {"time": time, "effect": effect}
            for time, effect in self.time_effects.items()
        ])

    @property
    def is_significant(self) -> bool:
        """Check if the ATT is statistically significant at the alpha level."""
        return bool(self.p_value < self.alpha)

    @property
    def significance_stars(self) -> str:
        """Return significance stars based on p-value."""
        return _get_significance_stars(self.p_value)


class TROP:
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
    method : str, default='twostep'
        Estimation method to use:

        - 'twostep': Per-observation model fitting following Algorithm 2 of
          Athey et al. (2025). Computes observation-specific weights and fits
          a model for each treated observation, averaging the individual
          treatment effects. More flexible but computationally intensive.

        - 'joint': Joint weighted least squares optimization. Estimates a
          single scalar treatment effect τ along with fixed effects and
          optional low-rank factor adjustment. Faster but assumes homogeneous
          treatment effects. Uses alternating minimization when nuclear norm
          penalty is finite.

    lambda_time_grid : list, optional
        Grid of time weight decay parameters. Default: [0, 0.1, 0.5, 1, 2, 5].
    lambda_unit_grid : list, optional
        Grid of unit weight decay parameters. Default: [0, 0.1, 0.5, 1, 2, 5].
    lambda_nn_grid : list, optional
        Grid of nuclear norm regularization parameters. Default: [0, 0.01, 0.1, 1].
    max_iter : int, default=100
        Maximum iterations for nuclear norm optimization.
    tol : float, default=1e-6
        Convergence tolerance for optimization.
    alpha : float, default=0.05
        Significance level for confidence intervals.
    variance_method : str, default='bootstrap'
        Method for variance estimation: 'bootstrap' or 'jackknife'.
    n_bootstrap : int, default=200
        Number of replications for variance estimation.
    max_loocv_samples : int, default=100
        Maximum control observations to use in LOOCV for tuning parameter
        selection. Subsampling is used for computational tractability as
        noted in the paper. Increase for more precise tuning at the cost
        of computational time.
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

    # Class constants
    DEFAULT_LOOCV_MAX_SAMPLES: int = 100
    """Maximum control observations to use in LOOCV (for computational tractability).

    As noted in the paper's footnote, LOOCV is subsampled for computational
    tractability. This constant controls the maximum number of control observations
    used in each LOOCV evaluation. Increase for more precise tuning at the cost
    of computational time.
    """

    CONVERGENCE_TOL_SVD: float = 1e-10
    """Tolerance for singular value truncation in soft-thresholding.

    Singular values below this threshold after soft-thresholding are treated
    as zero to improve numerical stability.
    """

    def __init__(
        self,
        method: str = "twostep",
        lambda_time_grid: Optional[List[float]] = None,
        lambda_unit_grid: Optional[List[float]] = None,
        lambda_nn_grid: Optional[List[float]] = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        alpha: float = 0.05,
        variance_method: str = 'bootstrap',
        n_bootstrap: int = 200,
        max_loocv_samples: int = 100,
        seed: Optional[int] = None,
    ):
        # Validate method parameter
        valid_methods = ("twostep", "joint")
        if method not in valid_methods:
            raise ValueError(
                f"method must be one of {valid_methods}, got '{method}'"
            )
        self.method = method

        # Default grids from paper
        self.lambda_time_grid = lambda_time_grid or [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
        self.lambda_unit_grid = lambda_unit_grid or [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
        self.lambda_nn_grid = lambda_nn_grid or [0.0, 0.01, 0.1, 1.0, 10.0]

        self.max_iter = max_iter
        self.tol = tol
        self.alpha = alpha
        self.variance_method = variance_method
        self.n_bootstrap = n_bootstrap
        self.max_loocv_samples = max_loocv_samples
        self.seed = seed

        # Validate parameters
        valid_variance_methods = ("bootstrap", "jackknife")
        if variance_method not in valid_variance_methods:
            raise ValueError(
                f"variance_method must be one of {valid_variance_methods}, "
                f"got '{variance_method}'"
            )

        # Internal state
        self.results_: Optional[TROPResults] = None
        self.is_fitted_: bool = False
        self._optimal_lambda: Optional[Tuple[float, float, float]] = None

        # Pre-computed structures (set during fit)
        self._precomputed: Optional[_PrecomputedStructures] = None

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
        control_obs = [(t, i) for t in range(n_periods) for i in range(n_units)
                       if control_mask[t, i] and not np.isnan(Y[t, i])]

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
        dist_unit_{-t}(j, i) = sqrt(Σ_u (Y_{iu} - Y_{ju})² / n_valid)

        For efficiency, we compute a base distance matrix excluding all treated
        observations, which provides a good approximation. The exact per-observation
        distances are refined when needed.

        Uses vectorized numpy operations with masked arrays for O(n²) complexity
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
        sq_diff = diff ** 2

        # Count valid (non-NaN) observations per pair
        # A difference is valid only if both units have valid observations
        valid_diff = ~np.isnan(sq_diff)
        n_valid = np.sum(valid_diff, axis=2)  # (n_units, n_units)

        # Compute sum of squared differences (treating NaN as 0)
        sq_diff_sum = np.nansum(sq_diff, axis=2)  # (n_units, n_units)

        # Compute RMSE distance: sqrt(sum / n_valid)
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
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
        can include _LAMBDA_INF values to disable specific components:
        - lambda_nn = inf: Skip nuclear norm regularization (L=0)
        - lambda_time = inf: Uniform time weights (treated as 0)
        - lambda_unit = inf: Uniform unit weights (treated as 0)

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
            Fixed values for other parameters. May include _LAMBDA_INF.

        Returns
        -------
        Tuple[float, float]
            (best_value, best_score) for the searched parameter.
        """
        best_score = np.inf
        best_value = grid[0] if grid else 0.0

        for value in grid:
            params = {**fixed_params, param_name: value}

            # Convert inf values to 0 for computation (inf means "disabled" = uniform weights)
            lambda_time = params.get('lambda_time', 0.0)
            lambda_unit = params.get('lambda_unit', 0.0)
            lambda_nn = params.get('lambda_nn', 0.0)

            # Handle infinity as "disabled" mode
            # Per paper Equations 2-3:
            # - λ_time/λ_unit=∞ → exp(-∞×dist)→0 for dist>0, uniform weights → use 0.0
            # - λ_nn=∞ → infinite penalty → L≈0 (factor model disabled) → use 1e10
            # Note: λ_nn=0 means NO regularization (full-rank L), opposite of "disabled"
            if np.isinf(lambda_time):
                lambda_time = 0.0  # Uniform time weights
            if np.isinf(lambda_unit):
                lambda_unit = 0.0  # Uniform unit weights
            if np.isinf(lambda_nn):
                lambda_nn = 1e10  # Very large → L≈0 (factor model disabled)

            try:
                score = self._loocv_score_obs_specific(
                    Y, D, control_mask, control_unit_idx,
                    lambda_time, lambda_unit, lambda_nn,
                    n_units, n_periods
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
                Y, D, control_mask, control_unit_idx, n_units, n_periods,
                'lambda_unit', self.lambda_unit_grid,
                {'lambda_time': lambda_time, 'lambda_nn': lambda_nn}
            )

            # Optimize λ_time (fix λ_unit, λ_nn)
            lambda_time, _ = self._univariate_loocv_search(
                Y, D, control_mask, control_unit_idx, n_units, n_periods,
                'lambda_time', self.lambda_time_grid,
                {'lambda_unit': lambda_unit, 'lambda_nn': lambda_nn}
            )

            # Optimize λ_nn (fix λ_unit, λ_time)
            lambda_nn, score = self._univariate_loocv_search(
                Y, D, control_mask, control_unit_idx, n_units, n_periods,
                'lambda_nn', self.lambda_nn_grid,
                {'lambda_unit': lambda_unit, 'lambda_time': lambda_time}
            )

            # Check convergence
            if abs(score - prev_score) < 1e-6:
                logger.debug(
                    "Cycling search converged after %d cycles with score %.6f",
                    cycle + 1, score
                )
                break
            prev_score = score

        return lambda_time, lambda_unit, lambda_nn

    # =========================================================================
    # Joint estimation method
    # =========================================================================

    def _compute_joint_weights(
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
        Compute distance-based weights for joint estimation.

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
        diff_sq = np.where(np.isfinite(diff), diff ** 2, 0.0) * pre_mask[:, np.newaxis]

        # Count valid observations per unit in pre-period
        valid_count = np.sum(
            np.isfinite(Y) * pre_mask[:, np.newaxis], axis=0
        )
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

        return delta

    def _loocv_score_joint(
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
        Compute LOOCV score for joint method with specific parameter combination.

        Following paper's Equation 5:
        Q(λ) = Σ_{j,s: D_js=0} [τ̂_js^loocv(λ)]²

        For joint method, we exclude each control observation, fit the joint model
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
        delta = self._compute_joint_weights(
            Y, D, lambda_time, lambda_unit, treated_periods, n_units, n_periods
        )

        tau_sq_sum = 0.0
        n_valid = 0

        for t_ex, i_ex in control_obs:
            # Create modified delta with excluded observation zeroed out
            delta_ex = delta.copy()
            delta_ex[t_ex, i_ex] = 0.0

            try:
                # Fit joint model excluding this observation
                if lambda_nn >= 1e10:
                    mu, alpha, beta, tau = self._solve_joint_no_lowrank(Y, D, delta_ex)
                    L = np.zeros((n_periods, n_units))
                else:
                    mu, alpha, beta, L, tau = self._solve_joint_with_lowrank(
                        Y, D, delta_ex, lambda_nn, self.max_iter, self.tol
                    )

                # Pseudo treatment effect: τ = Y - μ - α - β - L
                if np.isfinite(Y[t_ex, i_ex]):
                    tau_loocv = Y[t_ex, i_ex] - mu - alpha[i_ex] - beta[t_ex] - L[t_ex, i_ex]
                    tau_sq_sum += tau_loocv ** 2
                    n_valid += 1

            except (np.linalg.LinAlgError, ValueError):
                # Any failure means this λ combination is invalid per Equation 5
                return np.inf

        if n_valid == 0:
            return np.inf

        return tau_sq_sum

    def _solve_joint_no_lowrank(
        self,
        Y: np.ndarray,
        D: np.ndarray,
        delta: np.ndarray,
    ) -> Tuple[float, np.ndarray, np.ndarray, float]:
        """
        Solve joint TWFE + treatment via weighted least squares (no low-rank).

        Solves: min Σ δ_{it}(Y_{it} - μ - α_i - β_t - τ*W_{it})²

        Parameters
        ----------
        Y : np.ndarray
            Outcome matrix (n_periods x n_units).
        D : np.ndarray
            Treatment indicator matrix (n_periods x n_units).
        delta : np.ndarray
            Weight matrix (n_periods x n_units).

        Returns
        -------
        Tuple[float, np.ndarray, np.ndarray, float]
            (mu, alpha, beta, tau) estimated parameters.
        """
        n_periods, n_units = Y.shape

        # Flatten matrices for regression
        y = Y.flatten()  # length n_periods * n_units
        w = D.flatten()
        weights = delta.flatten()

        # Handle NaN values: zero weight for NaN outcomes/weights, impute with 0
        # This ensures NaN observations don't contribute to estimation
        valid_y = np.isfinite(y)
        valid_w = np.isfinite(weights)
        valid_mask = valid_y & valid_w
        weights = np.where(valid_mask, weights, 0.0)
        y = np.where(valid_mask, y, 0.0)

        sqrt_weights = np.sqrt(np.maximum(weights, 0))

        # Build design matrix: [intercept, unit_dummies, time_dummies, treatment]
        # Total columns: 1 + n_units + n_periods + 1
        # But we need to drop one unit and one time dummy for identification
        # Drop first unit (unit 0) and first time (time 0)
        n_obs = n_periods * n_units
        n_params = 1 + (n_units - 1) + (n_periods - 1) + 1

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

        # Treatment indicator
        X[:, -1] = w

        # Apply weights
        X_weighted = X * sqrt_weights[:, np.newaxis]
        y_weighted = y * sqrt_weights

        # Solve weighted least squares
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X_weighted, y_weighted, rcond=None)
        except np.linalg.LinAlgError:
            # Fallback: use pseudo-inverse
            coeffs = np.linalg.pinv(X_weighted) @ y_weighted

        # Extract parameters
        mu = coeffs[0]
        alpha = np.zeros(n_units)
        alpha[1:] = coeffs[1:n_units]
        beta = np.zeros(n_periods)
        beta[1:] = coeffs[n_units:(n_units + n_periods - 1)]
        tau = coeffs[-1]

        return float(mu), alpha, beta, float(tau)

    def _solve_joint_with_lowrank(
        self,
        Y: np.ndarray,
        D: np.ndarray,
        delta: np.ndarray,
        lambda_nn: float,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Solve joint TWFE + treatment + low-rank via alternating minimization.

        Solves: min Σ δ_{it}(Y_{it} - μ - α_i - β_t - L_{it} - τ*W_{it})² + λ_nn||L||_*

        Parameters
        ----------
        Y : np.ndarray
            Outcome matrix (n_periods x n_units).
        D : np.ndarray
            Treatment indicator matrix (n_periods x n_units).
        delta : np.ndarray
            Weight matrix (n_periods x n_units).
        lambda_nn : float
            Nuclear norm regularization parameter.
        max_iter : int, default=100
            Maximum iterations for alternating minimization.
        tol : float, default=1e-6
            Convergence tolerance.

        Returns
        -------
        Tuple[float, np.ndarray, np.ndarray, np.ndarray, float]
            (mu, alpha, beta, L, tau) estimated parameters.
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

        # Initialize L = 0
        L = np.zeros((n_periods, n_units))

        for iteration in range(max_iter):
            L_old = L.copy()

            # Step 1: Fix L, solve for (mu, alpha, beta, tau)
            # Adjusted outcome: Y - L (using NaN-safe Y)
            # Pass masked delta to exclude NaN observations from WLS
            Y_adj = Y_safe - L
            mu, alpha, beta, tau = self._solve_joint_no_lowrank(Y_adj, D, delta_masked)

            # Step 2: Fix (mu, alpha, beta, tau), update L
            # Residual: R = Y - mu - alpha - beta - tau*D (using NaN-safe Y)
            R = Y_safe - mu - alpha[np.newaxis, :] - beta[:, np.newaxis] - tau * D

            # Weighted proximal step for L (soft-threshold SVD)
            # Normalize weights (using masked delta to exclude NaN observations)
            delta_max = np.max(delta_masked)
            if delta_max > 0:
                delta_norm = delta_masked / delta_max
            else:
                delta_norm = delta_masked

            # Weighted average between current L and target R
            # L_next = L + delta_norm * (R - L), then soft-threshold
            # NaN observations have delta_norm=0, so they don't influence L update
            gradient_step = L + delta_norm * (R - L)

            # Soft-threshold singular values
            # Use eta * lambda_nn for proper proximal step size (matches Rust)
            eta = 1.0 / delta_max if delta_max > 0 else 1.0
            L = self._soft_threshold_svd(gradient_step, eta * lambda_nn)

            # Check convergence
            if np.max(np.abs(L - L_old)) < tol:
                break

        return mu, alpha, beta, L, tau

    def _fit_joint(
        self,
        data: pd.DataFrame,
        outcome: str,
        treatment: str,
        unit: str,
        time: str,
    ) -> TROPResults:
        """
        Fit TROP using joint weighted least squares method.

        This method estimates a single scalar treatment effect τ along with
        fixed effects and optional low-rank factor adjustment.

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
        Bootstrap and jackknife variance estimation assume simultaneous treatment
        adoption (fixed `treated_periods` across resamples). The treatment timing
        is inferred from the data once and held constant for all bootstrap/jackknife
        iterations. For staggered adoption designs where treatment timing varies
        across units, use `method="twostep"` which computes observation-specific
        weights that naturally handle heterogeneous timing.
        """
        # Data setup (same as twostep method)
        all_units = sorted(data[unit].unique())
        all_periods = sorted(data[time].unique())

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

        D_raw = (
            data.pivot(index=time, columns=unit, values=treatment)
            .reindex(index=all_periods, columns=all_units)
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
                f"D[t, unit] must be monotonic non-decreasing."
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

        # LOOCV grid search for tuning parameters
        # Use Rust backend when available for parallel LOOCV (5-10x speedup)
        best_lambda = None
        best_score = np.inf
        control_mask = D == 0

        if HAS_RUST_BACKEND and _rust_loocv_grid_search_joint is not None:
            try:
                # Prepare inputs for Rust function
                control_mask_u8 = control_mask.astype(np.uint8)

                lambda_time_arr = np.array(self.lambda_time_grid, dtype=np.float64)
                lambda_unit_arr = np.array(self.lambda_unit_grid, dtype=np.float64)
                lambda_nn_arr = np.array(self.lambda_nn_grid, dtype=np.float64)

                result = _rust_loocv_grid_search_joint(
                    Y, D.astype(np.float64), control_mask_u8,
                    lambda_time_arr, lambda_unit_arr, lambda_nn_arr,
                    self.max_loocv_samples, self.max_iter, self.tol,
                    self.seed if self.seed is not None else 0
                )
                # Unpack result - 7 values including optional first_failed_obs
                best_lt, best_lu, best_ln, best_score, n_valid, n_attempted, first_failed_obs = result
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
                        f"λ=({best_lt}, {best_lu}, {best_ln}). "
                        f"Returning infinite score.{obs_info}",
                        UserWarning
                    )
                elif n_attempted > 0 and (n_attempted - n_valid) > 0.1 * n_attempted:
                    n_failed = n_attempted - n_valid
                    obs_info = ""
                    if first_failed_obs is not None:
                        t_idx, i_idx = first_failed_obs
                        obs_info = f" First failure at observation ({t_idx}, {i_idx})."
                    warnings.warn(
                        f"LOOCV: {n_failed}/{n_attempted} fits failed for "
                        f"λ=({best_lt}, {best_lu}, {best_ln}). "
                        f"This may indicate numerical instability.{obs_info}",
                        UserWarning
                    )
            except Exception as e:
                # Fall back to Python implementation on error
                logger.debug(
                    "Rust LOOCV grid search (joint) failed, falling back to Python: %s", e
                )
                best_lambda = None
                best_score = np.inf

        # Fall back to Python implementation if Rust unavailable or failed
        if best_lambda is None:
            # Get control observations for LOOCV
            control_obs = [
                (t, i) for t in range(n_periods) for i in range(n_units)
                if control_mask[t, i] and not np.isnan(Y[t, i])
            ]

            # Subsample if needed
            if len(control_obs) > self.max_loocv_samples:
                rng = np.random.default_rng(self.seed)
                control_obs = list(
                    rng.choice(control_obs, size=self.max_loocv_samples, replace=False)
                )

            # Grid search with true LOOCV
            for lambda_time_val in self.lambda_time_grid:
                for lambda_unit_val in self.lambda_unit_grid:
                    for lambda_nn_val in self.lambda_nn_grid:
                        # Convert infinity values
                        lt = 0.0 if np.isinf(lambda_time_val) else lambda_time_val
                        lu = 0.0 if np.isinf(lambda_unit_val) else lambda_unit_val
                        ln = 1e10 if np.isinf(lambda_nn_val) else lambda_nn_val

                        try:
                            score = self._loocv_score_joint(
                                Y, D, control_obs, lt, lu, ln,
                                treated_periods, n_units, n_periods
                            )

                            if score < best_score:
                                best_score = score
                                best_lambda = (lambda_time_val, lambda_unit_val, lambda_nn_val)

                        except (np.linalg.LinAlgError, ValueError):
                            continue

        if best_lambda is None:
            warnings.warn(
                "All tuning parameter combinations failed. Using defaults.",
                UserWarning
            )
            best_lambda = (1.0, 1.0, 0.1)
            best_score = np.nan

        # Final estimation with best parameters
        lambda_time, lambda_unit, lambda_nn = best_lambda
        original_lambda_time, original_lambda_unit, original_lambda_nn = best_lambda

        # Convert infinity values for computation
        if np.isinf(lambda_time):
            lambda_time = 0.0
        if np.isinf(lambda_unit):
            lambda_unit = 0.0
        if np.isinf(lambda_nn):
            lambda_nn = 1e10

        # Compute final weights and fit
        delta = self._compute_joint_weights(
            Y, D, lambda_time, lambda_unit, treated_periods, n_units, n_periods
        )

        if lambda_nn >= 1e10:
            mu, alpha, beta, tau = self._solve_joint_no_lowrank(Y, D, delta)
            L = np.zeros((n_periods, n_units))
        else:
            mu, alpha, beta, L, tau = self._solve_joint_with_lowrank(
                Y, D, delta, lambda_nn, self.max_iter, self.tol
            )

        # ATT is the scalar treatment effect
        att = tau

        # Compute individual treatment effects for reporting (same τ for all)
        treatment_effects = {}
        for t in range(n_periods):
            for i in range(n_units):
                if D[t, i] == 1:
                    unit_id = idx_to_unit[i]
                    time_id = idx_to_period[t]
                    treatment_effects[(unit_id, time_id)] = tau

        # Compute effective rank of L
        _, s, _ = np.linalg.svd(L, full_matrices=False)
        if s[0] > 0:
            effective_rank = np.sum(s) / s[0]
        else:
            effective_rank = 0.0

        # Bootstrap variance estimation
        effective_lambda = (lambda_time, lambda_unit, lambda_nn)

        if self.variance_method == "bootstrap":
            se, bootstrap_dist = self._bootstrap_variance_joint(
                data, outcome, treatment, unit, time,
                effective_lambda, treated_periods
            )
        else:
            # Jackknife for joint method
            se, bootstrap_dist = self._jackknife_variance_joint(
                Y, D, effective_lambda, treated_periods,
                n_units, n_periods
            )

        # Compute test statistics
        if se > 0:
            t_stat = att / se
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=max(1, n_treated_obs - 1)))
            conf_int = compute_confidence_interval(att, se, self.alpha)
        else:
            t_stat = np.nan
            p_value = np.nan
            conf_int = (np.nan, np.nan)

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
            n_treated_obs=int(n_treated_obs),
            unit_effects=unit_effects_dict,
            time_effects=time_effects_dict,
            treatment_effects=treatment_effects,
            lambda_time=original_lambda_time,
            lambda_unit=original_lambda_unit,
            lambda_nn=original_lambda_nn,
            factor_matrix=L,
            effective_rank=effective_rank,
            loocv_score=best_score,
            variance_method=self.variance_method,
            alpha=self.alpha,
            n_pre_periods=n_pre_periods,
            n_post_periods=n_post_periods,
            n_bootstrap=self.n_bootstrap if self.variance_method == "bootstrap" else None,
            bootstrap_distribution=bootstrap_dist if len(bootstrap_dist) > 0 else None,
        )

        self.is_fitted_ = True
        return self.results_

    def _bootstrap_variance_joint(
        self,
        data: pd.DataFrame,
        outcome: str,
        treatment: str,
        unit: str,
        time: str,
        optimal_lambda: Tuple[float, float, float],
        treated_periods: int,
    ) -> Tuple[float, np.ndarray]:
        """
        Compute bootstrap standard error for joint method.

        Uses Rust backend when available for parallel bootstrap (5-15x speedup).

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

        Returns
        -------
        Tuple[float, np.ndarray]
            (se, bootstrap_estimates).
        """
        lambda_time, lambda_unit, lambda_nn = optimal_lambda

        # Try Rust backend for parallel bootstrap (5-15x speedup)
        if HAS_RUST_BACKEND and _rust_bootstrap_trop_variance_joint is not None:
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

                bootstrap_estimates, se = _rust_bootstrap_trop_variance_joint(
                    Y, D,
                    lambda_time, lambda_unit, lambda_nn,
                    self.n_bootstrap, self.max_iter, self.tol,
                    self.seed if self.seed is not None else 0
                )

                if len(bootstrap_estimates) < 10:
                    warnings.warn(
                        f"Only {len(bootstrap_estimates)} bootstrap iterations succeeded.",
                        UserWarning
                    )
                    if len(bootstrap_estimates) == 0:
                        return 0.0, np.array([])

                return float(se), np.array(bootstrap_estimates)

            except Exception as e:
                logger.debug(
                    "Rust bootstrap (joint) failed, falling back to Python: %s", e
                )

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
                sampled_control = rng.choice(
                    control_units, size=n_control_units, replace=True
                )
            else:
                sampled_control = np.array([], dtype=object)

            if n_treated_units > 0:
                sampled_treated = rng.choice(
                    treated_units, size=n_treated_units, replace=True
                )
            else:
                sampled_treated = np.array([], dtype=object)

            sampled_units = np.concatenate([sampled_control, sampled_treated])

            # Create bootstrap sample
            boot_data = pd.concat([
                data[data[unit] == u].assign(**{unit: f"{u}_{idx}"})
                for idx, u in enumerate(sampled_units)
            ], ignore_index=True)

            try:
                tau = self._fit_joint_with_fixed_lambda(
                    boot_data, outcome, treatment, unit, time,
                    optimal_lambda, treated_periods
                )
                bootstrap_estimates_list.append(tau)
            except (ValueError, np.linalg.LinAlgError, KeyError):
                continue

        bootstrap_estimates = np.array(bootstrap_estimates_list)

        if len(bootstrap_estimates) < 10:
            warnings.warn(
                f"Only {len(bootstrap_estimates)} bootstrap iterations succeeded.",
                UserWarning
            )
            if len(bootstrap_estimates) == 0:
                return 0.0, np.array([])

        se = np.std(bootstrap_estimates, ddof=1)
        return float(se), bootstrap_estimates

    def _fit_joint_with_fixed_lambda(
        self,
        data: pd.DataFrame,
        outcome: str,
        treatment: str,
        unit: str,
        time: str,
        fixed_lambda: Tuple[float, float, float],
        treated_periods: int,
    ) -> float:
        """
        Fit joint model with fixed tuning parameters.

        Returns only the treatment effect τ.
        """
        lambda_time, lambda_unit, lambda_nn = fixed_lambda

        all_units = sorted(data[unit].unique())
        all_periods = sorted(data[time].unique())

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

        # Compute weights
        delta = self._compute_joint_weights(
            Y, D, lambda_time, lambda_unit, treated_periods, n_units, n_periods
        )

        # Fit model
        if lambda_nn >= 1e10:
            _, _, _, tau = self._solve_joint_no_lowrank(Y, D, delta)
        else:
            _, _, _, _, tau = self._solve_joint_with_lowrank(
                Y, D, delta, lambda_nn, self.max_iter, self.tol
            )

        return tau

    def _jackknife_variance_joint(
        self,
        Y: np.ndarray,
        D: np.ndarray,
        optimal_lambda: Tuple[float, float, float],
        treated_periods: int,
        n_units: int,
        n_periods: int,
    ) -> Tuple[float, np.ndarray]:
        """
        Compute jackknife standard error for joint method.

        Parameters
        ----------
        Y : np.ndarray
            Outcome matrix.
        D : np.ndarray
            Treatment matrix.
        optimal_lambda : tuple
            Optimal tuning parameters.
        treated_periods : int
            Number of post-treatment periods.
        n_units : int
            Number of units.
        n_periods : int
            Number of periods.

        Returns
        -------
        Tuple[float, np.ndarray]
            (se, jackknife_estimates).
        """
        lambda_time, lambda_unit, lambda_nn = optimal_lambda
        jackknife_estimates = []

        # Get treated unit indices
        treated_unit_idx = np.where(np.any(D == 1, axis=0))[0]

        for leave_out in treated_unit_idx:
            # True leave-one-out: zero the delta weight for the left-out unit
            # This excludes the unit from estimation without imputation
            Y_jack = Y.copy()
            D_jack = D.copy()
            D_jack[:, leave_out] = 0  # Mark as not treated for weight computation

            try:
                # Compute weights (left-out unit is still in calculation)
                delta = self._compute_joint_weights(
                    Y_jack, D_jack, lambda_time, lambda_unit,
                    treated_periods, n_units, n_periods
                )

                # Zero the delta weight for the left-out unit
                # This ensures the unit doesn't contribute to estimation
                delta[:, leave_out] = 0.0

                # Fit model (left-out unit has zero weight, truly excluded)
                if lambda_nn >= 1e10:
                    _, _, _, tau = self._solve_joint_no_lowrank(Y_jack, D_jack, delta)
                else:
                    _, _, _, _, tau = self._solve_joint_with_lowrank(
                        Y_jack, D_jack, delta, lambda_nn, self.max_iter, self.tol
                    )

                jackknife_estimates.append(tau)

            except (np.linalg.LinAlgError, ValueError):
                continue

        jackknife_estimates = np.array(jackknife_estimates)

        if len(jackknife_estimates) < 2:
            return 0.0, jackknife_estimates

        # Jackknife SE formula
        n = len(jackknife_estimates)
        mean_est = np.mean(jackknife_estimates)
        se = np.sqrt((n - 1) / n * np.sum((jackknife_estimates - mean_est) ** 2))

        return float(se), jackknife_estimates

    def fit(
        self,
        data: pd.DataFrame,
        outcome: str,
        treatment: str,
        unit: str,
        time: str,
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

        Returns
        -------
        TROPResults
            Object containing the ATT estimate, standard error,
            factor estimates, and tuning parameters. The lambda_*
            attributes show the selected grid values. Infinity values
            (∞) are converted internally: λ_time/λ_unit=∞ → 0.0 (uniform
            weights), λ_nn=∞ → 1e10 (factor model disabled).
        """
        # Validate inputs
        required_cols = [outcome, treatment, unit, time]
        missing = [c for c in required_cols if c not in data.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # Dispatch based on estimation method
        if self.method == "joint":
            return self._fit_joint(data, outcome, treatment, unit, time)

        # Below is the twostep method (default)
        # Get unique units and periods
        all_units = sorted(data[unit].unique())
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
        D_raw = (
            data.pivot(index=time, columns=unit, values=treatment)
            .reindex(index=all_periods, columns=all_units)
        )
        missing_mask = pd.isna(D_raw).values  # True where originally missing
        D = D_raw.fillna(0).astype(int).values

        # Validate D is monotonic non-decreasing per unit (absorbing state)
        # D[t, i] must satisfy: once D=1, it must stay 1 for all subsequent periods
        # Issue 3 fix (round 10): Check each unit's OBSERVED D sequence for monotonicity
        # This catches 1→0 violations that span missing period gaps
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
        self._precomputed = self._precompute_structures(
            Y, D, control_unit_idx, n_units, n_periods
        )

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
                    Y, D.astype(np.float64), control_mask_u8,
                    time_dist_matrix,
                    lambda_time_arr, lambda_unit_arr, lambda_nn_arr,
                    self.max_loocv_samples, self.max_iter, self.tol,
                    self.seed if self.seed is not None else 0
                )
                # Unpack result - 7 values including optional first_failed_obs
                best_lt, best_lu, best_ln, best_score, n_valid, n_attempted, first_failed_obs = result
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
                        f"λ=({best_lt}, {best_lu}, {best_ln}). "
                        f"Returning infinite score.{obs_info}",
                        UserWarning
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
                        f"λ=({best_lt}, {best_lu}, {best_ln}). "
                        f"This may indicate numerical instability.{obs_info}",
                        UserWarning
                    )
            except Exception as e:
                # Fall back to Python implementation on error
                logger.debug(
                    "Rust LOOCV grid search failed, falling back to Python: %s", e
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
                Y, D, control_mask, control_unit_idx, n_units, n_periods,
                'lambda_time', self.lambda_time_grid,
                {'lambda_unit': 0.0, 'lambda_nn': _LAMBDA_INF}
            )

            # λ_nn search: fix λ_time=∞ (uniform time weights), λ_unit=0
            lambda_nn_init, _ = self._univariate_loocv_search(
                Y, D, control_mask, control_unit_idx, n_units, n_periods,
                'lambda_nn', self.lambda_nn_grid,
                {'lambda_time': _LAMBDA_INF, 'lambda_unit': 0.0}
            )

            # λ_unit search: fix λ_nn=∞, λ_time=0
            lambda_unit_init, _ = self._univariate_loocv_search(
                Y, D, control_mask, control_unit_idx, n_units, n_periods,
                'lambda_unit', self.lambda_unit_grid,
                {'lambda_nn': _LAMBDA_INF, 'lambda_time': 0.0}
            )

            # Stage 2: Cycling refinement (coordinate descent)
            lambda_time, lambda_unit, lambda_nn = self._cycling_parameter_search(
                Y, D, control_mask, control_unit_idx, n_units, n_periods,
                (lambda_time_init, lambda_unit_init, lambda_nn_init)
            )

            # Compute final score for the optimized parameters
            try:
                best_score = self._loocv_score_obs_specific(
                    Y, D, control_mask, control_unit_idx,
                    lambda_time, lambda_unit, lambda_nn,
                    n_units, n_periods
                )
                # Only accept finite scores - infinite means all fits failed
                if np.isfinite(best_score):
                    best_lambda = (lambda_time, lambda_unit, lambda_nn)
                # else: best_lambda stays None, triggering defaults fallback
            except (np.linalg.LinAlgError, ValueError):
                # If even the optimized parameters fail, best_lambda stays None
                pass

        if best_lambda is None:
            warnings.warn(
                "All tuning parameter combinations failed. Using defaults.",
                UserWarning
            )
            best_lambda = (1.0, 1.0, 0.1)
            best_score = np.nan

        self._optimal_lambda = best_lambda
        lambda_time, lambda_unit, lambda_nn = best_lambda

        # Convert infinity values for final estimation (matching LOOCV conversion)
        # This ensures final estimation uses the same effective parameters that LOOCV evaluated.
        # See REGISTRY.md "λ=∞ implementation" for rationale.
        #
        # IMPORTANT: Store original grid values for results, use converted for computation.
        # This lets users see what was selected from their grid, while ensuring consistent
        # behavior between point estimation and variance estimation.
        original_lambda_time, original_lambda_unit, original_lambda_nn = best_lambda

        if np.isinf(lambda_time):
            lambda_time = 0.0  # Uniform time weights
        if np.isinf(lambda_unit):
            lambda_unit = 0.0  # Uniform unit weights
        if np.isinf(lambda_nn):
            lambda_nn = 1e10  # Very large → L≈0 (factor model disabled)

        # Create effective_lambda with converted values for ALL downstream computation
        # This ensures variance estimation uses the same parameters as point estimation
        effective_lambda = (lambda_time, lambda_unit, lambda_nn)

        # Step 2: Final estimation - per-observation model fitting following Algorithm 2
        # For each treated (i,t): compute observation-specific weights, fit model, compute τ̂_{it}
        treatment_effects = {}
        tau_values = []
        alpha_estimates = []
        beta_estimates = []
        L_estimates = []

        # Use pre-computed treated observations
        treated_observations = self._precomputed["treated_observations"]

        for t, i in treated_observations:
            # Compute observation-specific weights for this (i, t)
            weight_matrix = self._compute_observation_weights(
                Y, D, i, t, lambda_time, lambda_unit, control_unit_idx,
                n_units, n_periods
            )

            # Fit model with these weights
            alpha_hat, beta_hat, L_hat = self._estimate_model(
                Y, control_mask, weight_matrix, lambda_nn,
                n_units, n_periods
            )

            # Compute treatment effect: τ̂_{it} = Y_{it} - α̂_i - β̂_t - L̂_{it}
            tau_it = Y[t, i] - alpha_hat[i] - beta_hat[t] - L_hat[t, i]

            unit_id = idx_to_unit[i]
            time_id = idx_to_period[t]
            treatment_effects[(unit_id, time_id)] = tau_it
            tau_values.append(tau_it)

            # Store for averaging
            alpha_estimates.append(alpha_hat)
            beta_estimates.append(beta_hat)
            L_estimates.append(L_hat)

        # Average ATT
        att = np.mean(tau_values)

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
        if self.variance_method == "bootstrap":
            se, bootstrap_dist = self._bootstrap_variance(
                data, outcome, treatment, unit, time,
                effective_lambda, Y=Y, D=D, control_unit_idx=control_unit_idx
            )
        else:
            se, bootstrap_dist = self._jackknife_variance(
                Y, D, control_mask, control_unit_idx, effective_lambda,
                n_units, n_periods
            )

        # Compute test statistics
        if se > 0:
            t_stat = att / se
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=max(1, n_treated_obs - 1)))
            conf_int = compute_confidence_interval(att, se, self.alpha)
        else:
            # When SE is undefined/zero, ALL inference fields should be NaN
            t_stat = np.nan
            p_value = np.nan
            conf_int = (np.nan, np.nan)

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
            n_treated_obs=n_treated_obs,
            unit_effects=unit_effects_dict,
            time_effects=time_effects_dict,
            treatment_effects=treatment_effects,
            # Store ORIGINAL grid values (possibly inf) so users see what was selected.
            # Internally, infinity values are converted for computation (see effective_lambda).
            lambda_time=original_lambda_time,
            lambda_unit=original_lambda_unit,
            lambda_nn=original_lambda_nn,
            factor_matrix=L_hat,
            effective_rank=effective_rank,
            loocv_score=best_score,
            variance_method=self.variance_method,
            alpha=self.alpha,
            n_pre_periods=n_pre_periods,
            n_post_periods=n_post_periods,
            n_bootstrap=self.n_bootstrap if self.variance_method == "bootstrap" else None,
            bootstrap_distribution=bootstrap_dist if len(bootstrap_dist) > 0 else None,
        )

        self.is_fitted_ = True
        return self.results_

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
        - Time weights θ_s^{i,t} = exp(-λ_time × |t - s|)
        - Unit weights ω_j^{i,t} = exp(-λ_unit × dist_unit_{-t}(j, i))

        IMPORTANT (Issue A fix): The paper's objective sums over ALL observations
        where (1 - W_js) is non-zero, which includes pre-treatment observations of
        eventually-treated units since W_js = 0 for those. This method computes
        weights for ALL units where D[t, j] = 0 at the target period, not just
        never-treated units.

        Uses pre-computed structures when available for efficiency.

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
        # Use pre-computed structures when available
        if self._precomputed is not None:
            # Time weights from pre-computed time distance matrix
            # time_dist_matrix[t, s] = |t - s|
            time_weights = np.exp(-lambda_time * self._precomputed["time_dist_matrix"][t, :])

            # Unit weights - computed for ALL units where D[t, j] = 0
            # (Issue A fix: includes pre-treatment obs of eventually-treated units)
            unit_weights = np.zeros(n_units)
            D_stored = self._precomputed["D"]
            Y_stored = self._precomputed["Y"]

            # Valid control units at time t: D[t, j] == 0
            valid_control_at_t = D_stored[t, :] == 0

            if lambda_unit == 0:
                # Uniform weights when lambda_unit = 0
                # All units not treated at time t get weight 1
                unit_weights[valid_control_at_t] = 1.0
            else:
                # Use observation-specific distances with target period excluded
                # (Issue B fix: compute exact per-observation distance)
                for j in range(n_units):
                    if valid_control_at_t[j] and j != i:
                        # Compute distance excluding target period t
                        dist = self._compute_unit_distance_for_obs(Y_stored, D_stored, j, i, t)
                        if np.isinf(dist):
                            unit_weights[j] = 0.0
                        else:
                            unit_weights[j] = np.exp(-lambda_unit * dist)

            # Treated unit i gets weight 1
            unit_weights[i] = 1.0

            # Weight matrix: outer product (n_periods x n_units)
            return np.outer(time_weights, unit_weights)

        # Fallback: compute from scratch (used in bootstrap/jackknife)
        # Time distance: |t - s| following paper's Equation 3 (page 7)
        dist_time = np.abs(np.arange(n_periods) - t)
        time_weights = np.exp(-lambda_time * dist_time)

        # Unit weights - computed for ALL units where D[t, j] = 0
        # (Issue A fix: includes pre-treatment obs of eventually-treated units)
        unit_weights = np.zeros(n_units)

        # Valid control units at time t: D[t, j] == 0
        valid_control_at_t = D[t, :] == 0

        if lambda_unit == 0:
            # Uniform weights when lambda_unit = 0
            unit_weights[valid_control_at_t] = 1.0
        else:
            for j in range(n_units):
                if valid_control_at_t[j] and j != i:
                    # Compute distance excluding target period t (Issue B fix)
                    dist = self._compute_unit_distance_for_obs(Y, D, j, i, t)
                    if np.isinf(dist):
                        unit_weights[j] = 0.0
                    else:
                        unit_weights[j] = np.exp(-lambda_unit * dist)

        # Treated unit i gets weight 1 (or could be omitted since we fit on controls)
        # We include treated unit's own observation for model fitting
        unit_weights[i] = 1.0

        # Weight matrix: outer product (n_periods x n_units)
        W = np.outer(time_weights, unit_weights)

        return W

    def _soft_threshold_svd(
        self,
        M: np.ndarray,
        threshold: float,
    ) -> np.ndarray:
        """
        Apply soft-thresholding to singular values (proximal operator for nuclear norm).

        Parameters
        ----------
        M : np.ndarray
            Input matrix.
        threshold : float
            Soft-thresholding parameter.

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
        nonzero_mask = s_thresh > self.CONVERGENCE_TOL_SVD
        if not np.any(nonzero_mask):
            return np.zeros_like(M)

        # Truncate to non-zero components for numerical stability
        U_trunc = U[:, nonzero_mask]
        s_trunc = s_thresh[nonzero_mask]
        Vt_trunc = Vt[nonzero_mask, :]

        # Compute result, suppressing expected numerical warnings from
        # ill-conditioned matrices during alternating minimization
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            result = (U_trunc * s_trunc) @ Vt_trunc

        # Replace any NaN/Inf in result with zeros
        if not np.isfinite(result).all():
            result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

        return result

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
            min_L Σ W_{ti}(R_{ti} - L_{ti})² + λ_nn||L||_*

        This uses a proximal gradient / soft-impute approach (Mazumder et al. 2010):
            L_{k+1} = prox_{λ||·||_*}(L_k + W ⊙ (R - L_k))

        where W ⊙ denotes element-wise multiplication with normalized weights.

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
        # Compute target residual R = Y - α - β
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

        # Proximal gradient iteration with weighted soft-impute
        # This solves: min_L ||W^{1/2} ⊙ (R - L)||_F^2 + λ||L||_*
        # Using: L_{k+1} = prox_{λ/η}(L_k + W ⊙ (R - L_k))
        # where η is the step size (we use η = 1 with normalized weights)
        for _ in range(max_inner_iter):
            L_old = L.copy()

            # Gradient step: L_k + W ⊙ (R - L_k)
            # For W=0 observations, this keeps L_k unchanged
            gradient_step = L + W_norm * (R_masked - L)

            # Proximal step: soft-threshold singular values
            L = self._soft_threshold_svd(gradient_step, lambda_nn)

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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate the model: Y = α + β + L + τD + ε with nuclear norm penalty on L.

        Uses alternating minimization with vectorized operations:
        1. Fix L, solve for α, β via weighted means
        2. Fix α, β, solve for L via soft-thresholding

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
        # Minimize: Σ W_{ti}(Y_{ti} - α_i - β_t - L_{ti})² + λ_nn||L||_*
        for _ in range(self.max_iter):
            alpha_old = alpha.copy()
            beta_old = beta.copy()
            L_old = L.copy()

            # Step 1: Update α and β (weighted least squares)
            # Following Equation 2 (page 7), fix L and solve for α, β
            # R = Y - L (residual without fixed effects)
            R = Y_safe - L

            # Alpha update (unit fixed effects):
            # α_i = argmin_α Σ_t W_{ti}(R_{ti} - α - β_t)²
            # Solution: α_i = Σ_t W_{ti}(R_{ti} - β_t) / Σ_t W_{ti}
            R_minus_beta = R - beta[:, np.newaxis]  # (n_periods, n_units)
            weighted_R_minus_beta = W_masked * R_minus_beta
            alpha_numerator = np.sum(weighted_R_minus_beta, axis=0)  # (n_units,)
            alpha = np.where(unit_has_obs, alpha_numerator / safe_unit_denom, 0.0)

            # Beta update (time fixed effects):
            # β_t = argmin_β Σ_i W_{ti}(R_{ti} - α_i - β)²
            # Solution: β_t = Σ_i W_{ti}(R_{ti} - α_i) / Σ_i W_{ti}
            R_minus_alpha = R - alpha[np.newaxis, :]  # (n_periods, n_units)
            weighted_R_minus_alpha = W_masked * R_minus_alpha
            beta_numerator = np.sum(weighted_R_minus_alpha, axis=1)  # (n_periods,)
            beta = np.where(time_has_obs, beta_numerator / safe_time_denom, 0.0)

            # Step 2: Update L with weighted nuclear norm penalty
            # Issue C fix: Use weighted soft-impute to properly account for
            # observation weights in the nuclear norm optimization.
            # Following Equation 2 (page 7): min_L Σ W_{ti}(Y - α - β - L)² + λ||L||_*
            L = self._weighted_nuclear_norm_solve(
                Y_safe, W_masked, L, alpha, beta, lambda_nn, max_inner_iter=10
            )

            # Check convergence
            alpha_diff = np.max(np.abs(alpha - alpha_old))
            beta_diff = np.max(np.abs(beta - beta_old))
            L_diff = np.max(np.abs(L - L_old))

            if max(alpha_diff, beta_diff, L_diff) < self.tol:
                break

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
        Q(λ) = Σ_{j,s: D_js=0} [τ̂_js^loocv(λ)]²

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
            control_obs = [(t, i) for t in range(n_periods) for i in range(n_units)
                           if control_mask[t, i] and not np.isnan(Y[t, i])]

        # Subsample for computational tractability (as noted in paper's footnote)
        rng = np.random.default_rng(self.seed)
        max_loocv = min(self.max_loocv_samples, len(control_obs))
        if len(control_obs) > max_loocv:
            indices = rng.choice(len(control_obs), size=max_loocv, replace=False)
            control_obs = [control_obs[idx] for idx in indices]

        # Empty control set check: if no control observations, return infinity
        # A score of 0.0 would incorrectly "win" over legitimate parameters
        if len(control_obs) == 0:
            warnings.warn(
                f"LOOCV: No valid control observations for "
                f"λ=({lambda_time}, {lambda_unit}, {lambda_nn}). "
                "Returning infinite score.",
                UserWarning
            )
            return np.inf

        tau_squared_sum = 0.0
        n_valid = 0

        for t, i in control_obs:
            try:
                # Compute observation-specific weights for pseudo-treated (i, t)
                # Uses pre-computed distance matrices when available
                weight_matrix = self._compute_observation_weights(
                    Y, D, i, t, lambda_time, lambda_unit, control_unit_idx,
                    n_units, n_periods
                )

                # Estimate model excluding observation (t, i)
                alpha, beta, L = self._estimate_model(
                    Y, control_mask, weight_matrix, lambda_nn,
                    n_units, n_periods, exclude_obs=(t, i)
                )

                # Pseudo treatment effect
                tau_ti = Y[t, i] - alpha[i] - beta[t] - L[t, i]
                tau_squared_sum += tau_ti ** 2
                n_valid += 1

            except (np.linalg.LinAlgError, ValueError):
                # Per Equation 5: Q(λ) must sum over ALL D==0 cells
                # Any failure means this λ cannot produce valid estimates for all cells
                warnings.warn(
                    f"LOOCV: Fit failed for observation ({t}, {i}) with "
                    f"λ=({lambda_time}, {lambda_unit}, {lambda_nn}). "
                    "Returning infinite score per Equation 5.",
                    UserWarning
                )
                return np.inf

        # Return SUM of squared pseudo-treatment effects per Equation 5 (page 8):
        # Q(λ) = Σ_{j,s: D_js=0} [τ̂_js^loocv(λ)]²
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
    ) -> Tuple[float, np.ndarray]:
        """
        Compute bootstrap standard error using unit-level block bootstrap.

        When the optional Rust backend is available and the matrix parameters
        (Y, D, control_unit_idx) are provided, uses parallelized Rust
        implementation for 5-15x speedup. Falls back to Python implementation
        if Rust is unavailable or if matrix parameters are not provided.

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

        # Try Rust backend for parallel bootstrap (5-15x speedup)
        if (HAS_RUST_BACKEND and _rust_bootstrap_trop_variance is not None
                and self._precomputed is not None and Y is not None
                and D is not None):
            try:
                control_mask = self._precomputed["control_mask"]
                time_dist_matrix = self._precomputed["time_dist_matrix"].astype(np.int64)

                bootstrap_estimates, se = _rust_bootstrap_trop_variance(
                    Y, D.astype(np.float64),
                    control_mask.astype(np.uint8),
                    time_dist_matrix,
                    lambda_time, lambda_unit, lambda_nn,
                    self.n_bootstrap, self.max_iter, self.tol,
                    self.seed if self.seed is not None else 0
                )

                if len(bootstrap_estimates) >= 10:
                    return float(se), bootstrap_estimates
                # Fall through to Python if too few bootstrap samples
                logger.debug(
                    "Rust bootstrap returned only %d samples, falling back to Python",
                    len(bootstrap_estimates)
                )
            except Exception as e:
                logger.debug(
                    "Rust bootstrap variance failed, falling back to Python: %s", e
                )

        # Python implementation (fallback)
        rng = np.random.default_rng(self.seed)

        # Issue D fix: Stratified bootstrap sampling
        # Paper's Algorithm 3 (page 27) specifies sampling N_0 control rows
        # and N_1 treated rows separately to preserve treatment ratio
        unit_ever_treated = data.groupby(unit)[treatment].max()
        treated_units = np.array(unit_ever_treated[unit_ever_treated == 1].index)
        control_units = np.array(unit_ever_treated[unit_ever_treated == 0].index)

        n_treated_units = len(treated_units)
        n_control_units = len(control_units)

        bootstrap_estimates_list = []

        for _ in range(self.n_bootstrap):
            # Stratified sampling: sample control and treated units separately
            # This preserves the treatment ratio in each bootstrap sample
            if n_control_units > 0:
                sampled_control = rng.choice(
                    control_units, size=n_control_units, replace=True
                )
            else:
                sampled_control = np.array([], dtype=control_units.dtype)

            if n_treated_units > 0:
                sampled_treated = rng.choice(
                    treated_units, size=n_treated_units, replace=True
                )
            else:
                sampled_treated = np.array([], dtype=treated_units.dtype)

            # Combine stratified samples
            sampled_units = np.concatenate([sampled_control, sampled_treated])

            # Create bootstrap sample with unique unit IDs
            boot_data = pd.concat([
                data[data[unit] == u].assign(**{unit: f"{u}_{idx}"})
                for idx, u in enumerate(sampled_units)
            ], ignore_index=True)

            try:
                # Fit with fixed lambda (skip LOOCV for speed)
                att = self._fit_with_fixed_lambda(
                    boot_data, outcome, treatment, unit, time,
                    optimal_lambda
                )
                bootstrap_estimates_list.append(att)
            except (ValueError, np.linalg.LinAlgError, KeyError):
                continue

        bootstrap_estimates = np.array(bootstrap_estimates_list)

        if len(bootstrap_estimates) < 10:
            warnings.warn(
                f"Only {len(bootstrap_estimates)} bootstrap iterations succeeded. "
                "Standard errors may be unreliable.",
                UserWarning
            )
            if len(bootstrap_estimates) == 0:
                return 0.0, np.array([])

        se = np.std(bootstrap_estimates, ddof=1)
        return float(se), bootstrap_estimates

    def _jackknife_variance(
        self,
        Y: np.ndarray,
        D: np.ndarray,
        control_mask: np.ndarray,
        control_unit_idx: np.ndarray,
        optimal_lambda: Tuple[float, float, float],
        n_units: int,
        n_periods: int,
    ) -> Tuple[float, np.ndarray]:
        """
        Compute jackknife standard error (leave-one-unit-out).

        Uses observation-specific weights following Algorithm 2.

        Parameters
        ----------
        Y : np.ndarray
            Outcome matrix.
        D : np.ndarray
            Treatment matrix.
        control_mask : np.ndarray
            Control observation mask.
        control_unit_idx : np.ndarray
            Indices of control units.
        optimal_lambda : tuple
            Optimal tuning parameters.
        n_units : int
            Number of units.
        n_periods : int
            Number of periods.

        Returns
        -------
        tuple
            (se, jackknife_estimates).
        """
        lambda_time, lambda_unit, lambda_nn = optimal_lambda
        jackknife_estimates = []

        # Get treated unit indices
        treated_unit_idx = np.where(np.any(D == 1, axis=0))[0]

        for leave_out in treated_unit_idx:
            # Create mask excluding this unit
            Y_jack = Y.copy()
            D_jack = D.copy()
            Y_jack[:, leave_out] = np.nan
            D_jack[:, leave_out] = 0

            control_mask_jack = D_jack == 0

            # Get remaining treated observations
            treated_obs_jack = [(t, i) for t in range(n_periods) for i in range(n_units)
                                if D_jack[t, i] == 1]

            if not treated_obs_jack:
                continue

            try:
                # Compute ATT using observation-specific weights (Algorithm 2)
                tau_values = []
                for t, i in treated_obs_jack:
                    # Compute observation-specific weights for this (i, t)
                    weight_matrix = self._compute_observation_weights(
                        Y_jack, D_jack, i, t, lambda_time, lambda_unit,
                        control_unit_idx, n_units, n_periods
                    )

                    # Fit model with these weights
                    alpha, beta, L = self._estimate_model(
                        Y_jack, control_mask_jack, weight_matrix, lambda_nn,
                        n_units, n_periods
                    )

                    # Compute treatment effect
                    tau = Y_jack[t, i] - alpha[i] - beta[t] - L[t, i]
                    tau_values.append(tau)

                if tau_values:
                    jackknife_estimates.append(np.mean(tau_values))

            except (np.linalg.LinAlgError, ValueError):
                continue

        jackknife_estimates = np.array(jackknife_estimates)

        if len(jackknife_estimates) < 2:
            return 0.0, jackknife_estimates

        # Jackknife SE formula
        n = len(jackknife_estimates)
        mean_est = np.mean(jackknife_estimates)
        se = np.sqrt((n - 1) / n * np.sum((jackknife_estimates - mean_est) ** 2))

        return se, jackknife_estimates

    def _fit_with_fixed_lambda(
        self,
        data: pd.DataFrame,
        outcome: str,
        treatment: str,
        unit: str,
        time: str,
        fixed_lambda: Tuple[float, float, float],
    ) -> float:
        """
        Fit model with fixed tuning parameters (for bootstrap).

        Uses observation-specific weights following Algorithm 2.
        Returns only the ATT estimate.
        """
        lambda_time, lambda_unit, lambda_nn = fixed_lambda

        # Setup matrices
        all_units = sorted(data[unit].unique())
        all_periods = sorted(data[time].unique())

        n_units = len(all_units)
        n_periods = len(all_periods)

        unit_to_idx = {u: i for i, u in enumerate(all_units)}
        period_to_idx = {p: i for i, p in enumerate(all_periods)}

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
        treated_observations = [(t, i) for t in range(n_periods) for i in range(n_units)
                                if D[t, i] == 1]

        if not treated_observations:
            raise ValueError("No treated observations")

        # Compute ATT using observation-specific weights (Algorithm 2)
        tau_values = []
        for t, i in treated_observations:
            # Compute observation-specific weights for this (i, t)
            weight_matrix = self._compute_observation_weights(
                Y, D, i, t, lambda_time, lambda_unit, control_unit_idx,
                n_units, n_periods
            )

            # Fit model with these weights
            alpha, beta, L = self._estimate_model(
                Y, control_mask, weight_matrix, lambda_nn,
                n_units, n_periods
            )

            # Compute treatment effect: τ̂_{it} = Y_{it} - α̂_i - β̂_t - L̂_{it}
            tau = Y[t, i] - alpha[i] - beta[t] - L[t, i]
            tau_values.append(tau)

        return np.mean(tau_values)

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
            "variance_method": self.variance_method,
            "n_bootstrap": self.n_bootstrap,
            "max_loocv_samples": self.max_loocv_samples,
            "seed": self.seed,
        }

    def set_params(self, **params) -> "TROP":
        """Set estimator parameters."""
        for key, value in params.items():
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
    return estimator.fit(data, outcome, treatment, unit, time)
