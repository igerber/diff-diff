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

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from diff_diff.results import _get_significance_stars
from diff_diff.utils import compute_confidence_interval, compute_p_value


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
        Selected time weight decay parameter.
    lambda_unit : float
        Selected unit weight decay parameter.
    lambda_nn : float
        Selected nuclear norm regularization parameter.
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
    pre_periods : list
        List of pre-treatment period identifiers.
    post_periods : list
        List of post-treatment period identifiers.
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
    pre_periods: List[Any] = field(default_factory=list)
    post_periods: List[Any] = field(default_factory=list)
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
            f"{'Pre-treatment periods:':<25} {len(self.pre_periods):>10}",
            f"{'Post-treatment periods:':<25} {len(self.post_periods):>10}",
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
            "n_pre_periods": len(self.pre_periods),
            "n_post_periods": len(self.post_periods),
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
    ...     post_periods=[5, 6, 7, 8]
    ... )
    >>> results.print_summary()

    References
    ----------
    Athey, S., Imbens, G. W., Qu, Z., & Viviano, D. (2025). Triply Robust
    Panel Estimators. *Working Paper*. https://arxiv.org/abs/2508.21536
    """

    def __init__(
        self,
        lambda_time_grid: Optional[List[float]] = None,
        lambda_unit_grid: Optional[List[float]] = None,
        lambda_nn_grid: Optional[List[float]] = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        alpha: float = 0.05,
        variance_method: str = 'bootstrap',
        n_bootstrap: int = 200,
        seed: Optional[int] = None,
    ):
        # Default grids from paper
        self.lambda_time_grid = lambda_time_grid or [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
        self.lambda_unit_grid = lambda_unit_grid or [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
        self.lambda_nn_grid = lambda_nn_grid or [0.0, 0.01, 0.1, 1.0, 10.0]

        self.max_iter = max_iter
        self.tol = tol
        self.alpha = alpha
        self.variance_method = variance_method
        self.n_bootstrap = n_bootstrap
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

    def fit(
        self,
        data: pd.DataFrame,
        outcome: str,
        treatment: str,
        unit: str,
        time: str,
        post_periods: Optional[List[Any]] = None,
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
            Should be 1 for treated unit-time observations.
        unit : str
            Name of the unit identifier column.
        time : str
            Name of the time period column.
        post_periods : list, optional
            List of time period values that are post-treatment.
            If None, infers from treatment indicator.

        Returns
        -------
        TROPResults
            Object containing the ATT estimate, standard error,
            factor estimates, and tuning parameters.
        """
        # Validate inputs
        required_cols = [outcome, treatment, unit, time]
        missing = [c for c in required_cols if c not in data.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

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
        Y = np.full((n_periods, n_units), np.nan)
        D = np.zeros((n_periods, n_units), dtype=int)

        for _, row in data.iterrows():
            i = unit_to_idx[row[unit]]
            t = period_to_idx[row[time]]
            Y[t, i] = row[outcome]
            D[t, i] = int(row[treatment])

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
        if post_periods is None:
            # Infer from first treatment time
            first_treat_period = None
            for t in range(n_periods):
                if np.any(D[t, :] == 1):
                    first_treat_period = t
                    break
            if first_treat_period is None:
                raise ValueError("Could not infer post-treatment periods")
            pre_period_idx = list(range(first_treat_period))
            post_period_idx = list(range(first_treat_period, n_periods))
        else:
            post_period_idx = [period_to_idx[p] for p in post_periods if p in period_to_idx]
            pre_period_idx = [i for i in range(n_periods) if i not in post_period_idx]

        if len(pre_period_idx) < 2:
            raise ValueError("Need at least 2 pre-treatment periods")

        pre_periods_list = [idx_to_period[i] for i in pre_period_idx]
        post_periods_list = [idx_to_period[i] for i in post_period_idx]
        n_treated_periods = len(post_period_idx)

        # Step 1: Grid search with LOOCV for tuning parameters
        best_lambda = None
        best_score = np.inf

        # Control observations mask (for LOOCV)
        control_mask = D == 0

        for lambda_time in self.lambda_time_grid:
            for lambda_unit in self.lambda_unit_grid:
                for lambda_nn in self.lambda_nn_grid:
                    try:
                        score = self._loocv_score_obs_specific(
                            Y, D, control_mask, control_unit_idx,
                            lambda_time, lambda_unit, lambda_nn,
                            n_units, n_periods
                        )
                        if score < best_score:
                            best_score = score
                            best_lambda = (lambda_time, lambda_unit, lambda_nn)
                    except (np.linalg.LinAlgError, ValueError):
                        continue

        if best_lambda is None:
            warnings.warn(
                "All tuning parameter combinations failed. Using defaults.",
                UserWarning
            )
            best_lambda = (1.0, 1.0, 0.1)
            best_score = np.nan

        self._optimal_lambda = best_lambda
        lambda_time, lambda_unit, lambda_nn = best_lambda

        # Step 2: Final estimation - per-observation model fitting following Algorithm 2
        # For each treated (i,t): compute observation-specific weights, fit model, compute τ̂_{it}
        treatment_effects = {}
        tau_values = []
        alpha_estimates = []
        beta_estimates = []
        L_estimates = []

        # Get list of treated observations
        treated_observations = [(t, i) for t in range(n_periods) for i in range(n_units)
                                if D[t, i] == 1]

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
        if self.variance_method == "bootstrap":
            se, bootstrap_dist = self._bootstrap_variance(
                data, outcome, treatment, unit, time, post_periods_list,
                best_lambda
            )
        else:
            se, bootstrap_dist = self._jackknife_variance(
                Y, D, control_mask, control_unit_idx, best_lambda,
                n_units, n_periods
            )

        # Compute test statistics
        if se > 0:
            t_stat = att / se
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=max(1, n_treated_obs - 1)))
        else:
            t_stat = 0.0
            p_value = 1.0

        conf_int = compute_confidence_interval(att, se, self.alpha)

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
            lambda_time=lambda_time,
            lambda_unit=lambda_unit,
            lambda_nn=lambda_nn,
            factor_matrix=L_hat,
            effective_rank=effective_rank,
            loocv_score=best_score,
            variance_method=self.variance_method,
            alpha=self.alpha,
            pre_periods=pre_periods_list,
            post_periods=post_periods_list,
            n_bootstrap=self.n_bootstrap if self.variance_method == "bootstrap" else None,
            bootstrap_distribution=bootstrap_dist if len(bootstrap_dist) > 0 else None,
        )

        self.is_fitted_ = True
        return self.results_

    def _compute_unit_distance_pairwise(
        self,
        Y: np.ndarray,
        D: np.ndarray,
        j: int,
        i: int,
        target_period: int,
    ) -> float:
        """
        Compute pairwise distance from control unit j to treated unit i.

        Following the paper's Equation 3 (page 7):
        dist_unit_{-t}(j, i) = sqrt(
            Σ_u 1{u≠t}(1-W_{iu})(1-W_{ju})(Y_{iu} - Y_{ju})²
            / Σ_u 1{u≠t}(1-W_{iu})(1-W_{ju})
        )

        This computes the RMSE between units j and i over periods where
        both are untreated, excluding the target period t.

        Parameters
        ----------
        Y : np.ndarray
            Outcome matrix (n_periods x n_units).
        D : np.ndarray
            Treatment indicator matrix (n_periods x n_units).
        j : int
            Index of control unit.
        i : int
            Index of treated unit.
        target_period : int
            Target treatment period t (excluded from distance computation).

        Returns
        -------
        float
            Pairwise RMSE distance between units j and i.
        """
        n_periods = Y.shape[0]

        sq_diffs = []
        for u in range(n_periods):
            # Exclude target period and periods where either unit is treated
            if u == target_period:
                continue
            # (1 - W_{iu})(1 - W_{ju}) means both must be untreated
            if D[u, i] == 1 or D[u, j] == 1:
                continue
            if np.isnan(Y[u, i]) or np.isnan(Y[u, j]):
                continue

            sq_diffs.append((Y[u, i] - Y[u, j]) ** 2)

        if len(sq_diffs) > 0:
            return np.sqrt(np.mean(sq_diffs))
        else:
            return np.inf

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

        Following the paper's Algorithm 2 (page 27):
        - Time weights θ_s^{i,t} = exp(-λ_time × |t - s|)
        - Unit weights ω_j^{i,t} = exp(-λ_unit × dist_unit_{-t}(j, i))

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
            Indices of control units.
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
        dist_time = np.array([abs(t - s) for s in range(n_periods)])
        time_weights = np.exp(-lambda_time * dist_time)

        # Unit distance: pairwise RMSE from each control j to treated i
        unit_weights = np.zeros(n_units)

        if lambda_unit == 0:
            # Uniform weights when lambda_unit = 0
            unit_weights[:] = 1.0
        else:
            for j in control_unit_idx:
                dist = self._compute_unit_distance_pairwise(Y, D, j, i, t)
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
        nonzero_mask = s_thresh > 1e-10
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

        Uses alternating minimization:
        1. Fix L, solve for α, β
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

        # Alternating minimization
        for iteration in range(self.max_iter):
            alpha_old = alpha.copy()
            beta_old = beta.copy()
            L_old = L.copy()

            # Step 1: Update α and β (weighted means)
            R = Y - L  # Residual without fixed effects

            # Weighted mean for alpha (unit effects)
            for i in range(n_units):
                mask_i = valid_mask[:, i]
                if np.any(mask_i):
                    weights_i = W[mask_i, i]
                    # Handle case where weights sum to zero (unit not in weight computation)
                    weight_sum = np.sum(weights_i)
                    if weight_sum > 0:
                        alpha[i] = np.average(R[mask_i, i] - beta[mask_i], weights=weights_i)
                    else:
                        # Use unweighted mean for units with zero total weight
                        alpha[i] = np.mean(R[mask_i, i] - beta[mask_i])
                else:
                    alpha[i] = 0.0

            # Weighted mean for beta (time effects)
            for t in range(n_periods):
                mask_t = valid_mask[t, :]
                if np.any(mask_t):
                    weights_t = W[t, mask_t]
                    # Handle case where weights sum to zero
                    weight_sum = np.sum(weights_t)
                    if weight_sum > 0:
                        beta[t] = np.average(R[t, mask_t] - alpha[mask_t], weights=weights_t)
                    else:
                        # Use unweighted mean for periods with zero total weight
                        beta[t] = np.mean(R[t, mask_t] - alpha[mask_t])
                else:
                    beta[t] = 0.0

            # Step 2: Update L with nuclear norm penalty
            # L = soft_threshold(Y - α - β, λ_nn)
            R_for_L = np.zeros((n_periods, n_units))
            for t in range(n_periods):
                for i in range(n_units):
                    if valid_mask[t, i]:
                        R_for_L[t, i] = Y[t, i] - alpha[i] - beta[t]
                    else:
                        # Impute with current L
                        R_for_L[t, i] = L[t, i]

            L = self._soft_threshold_svd(R_for_L, lambda_nn)

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
        # Get all control observations
        control_obs = [(t, i) for t in range(n_periods) for i in range(n_units)
                       if control_mask[t, i] and not np.isnan(Y[t, i])]

        # Subsample for computational tractability (as noted in paper's footnote)
        rng = np.random.default_rng(self.seed)
        max_loocv = min(100, len(control_obs))
        if len(control_obs) > max_loocv:
            indices = rng.choice(len(control_obs), size=max_loocv, replace=False)
            control_obs = [control_obs[idx] for idx in indices]

        tau_squared_sum = 0.0
        n_valid = 0

        for t, i in control_obs:
            try:
                # Compute observation-specific weights for pseudo-treated (i, t)
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
                continue

        if n_valid == 0:
            return np.inf

        return tau_squared_sum / n_valid

    def _bootstrap_variance(
        self,
        data: pd.DataFrame,
        outcome: str,
        treatment: str,
        unit: str,
        time: str,
        post_periods: List[Any],
        optimal_lambda: Tuple[float, float, float],
    ) -> Tuple[float, np.ndarray]:
        """
        Compute bootstrap standard error using unit-level block bootstrap.

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
        post_periods : list
            Post-treatment periods.
        optimal_lambda : tuple
            Optimal (lambda_time, lambda_unit, lambda_nn).

        Returns
        -------
        tuple
            (se, bootstrap_estimates).
        """
        rng = np.random.default_rng(self.seed)
        all_units = data[unit].unique()
        n_units = len(all_units)

        bootstrap_estimates = []

        for b in range(self.n_bootstrap):
            # Sample units with replacement
            sampled_units = rng.choice(all_units, size=n_units, replace=True)

            # Create bootstrap sample with unique unit IDs
            boot_data = pd.concat([
                data[data[unit] == u].assign(**{unit: f"{u}_{idx}"})
                for idx, u in enumerate(sampled_units)
            ], ignore_index=True)

            try:
                # Fit with fixed lambda (skip LOOCV for speed)
                att = self._fit_with_fixed_lambda(
                    boot_data, outcome, treatment, unit, time,
                    post_periods, optimal_lambda
                )
                bootstrap_estimates.append(att)
            except (ValueError, np.linalg.LinAlgError, KeyError):
                continue

        bootstrap_estimates = np.array(bootstrap_estimates)

        if len(bootstrap_estimates) < 10:
            warnings.warn(
                f"Only {len(bootstrap_estimates)} bootstrap iterations succeeded. "
                "Standard errors may be unreliable.",
                UserWarning
            )
            if len(bootstrap_estimates) == 0:
                return 0.0, np.array([])

        se = np.std(bootstrap_estimates, ddof=1)
        return se, bootstrap_estimates

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
        post_periods: List[Any],
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

        Y = np.full((n_periods, n_units), np.nan)
        D = np.zeros((n_periods, n_units), dtype=int)

        for _, row in data.iterrows():
            i = unit_to_idx[row[unit]]
            t = period_to_idx[row[time]]
            Y[t, i] = row[outcome]
            D[t, i] = int(row[treatment])

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
            "lambda_time_grid": self.lambda_time_grid,
            "lambda_unit_grid": self.lambda_unit_grid,
            "lambda_nn_grid": self.lambda_nn_grid,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "alpha": self.alpha,
            "variance_method": self.variance_method,
            "n_bootstrap": self.n_bootstrap,
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
    post_periods: Optional[List[Any]] = None,
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
        Treatment indicator column name.
    unit : str
        Unit identifier column name.
    time : str
        Time period column name.
    post_periods : list, optional
        Post-treatment periods.
    **kwargs
        Additional arguments passed to TROP constructor.

    Returns
    -------
    TROPResults
        Estimation results.

    Examples
    --------
    >>> from diff_diff import trop
    >>> results = trop(data, 'y', 'treated', 'unit', 'time', post_periods=[5,6,7])
    >>> print(f"ATT: {results.att:.3f}")
    """
    estimator = TROP(**kwargs)
    return estimator.fit(data, outcome, treatment, unit, time, post_periods)
