"""
Synthetic Difference-in-Differences estimator.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError

from diff_diff.estimators import DifferenceInDifferences
from diff_diff.results import SyntheticDiDResults
from diff_diff.utils import (
    compute_confidence_interval,
    compute_p_value,
    compute_sdid_estimator,
    compute_synthetic_weights,
    compute_time_weights,
    validate_binary,
)


class SyntheticDiD(DifferenceInDifferences):
    """
    Synthetic Difference-in-Differences (SDID) estimator.

    Combines the strengths of Difference-in-Differences and Synthetic Control
    methods by re-weighting control units to better match treated units'
    pre-treatment trends.

    This method is particularly useful when:
    - You have few treated units (possibly just one)
    - Parallel trends assumption may be questionable
    - Control units are heterogeneous and need reweighting
    - You want robustness to pre-treatment differences

    Parameters
    ----------
    lambda_reg : float, default=0.0
        L2 regularization for unit weights. Larger values shrink weights
        toward uniform. Useful when n_pre_periods < n_control_units.
    zeta : float, default=1.0
        Regularization for time weights. Larger values give more uniform
        time weights (closer to standard DiD).
    alpha : float, default=0.05
        Significance level for confidence intervals.
    variance_method : str, default="bootstrap"
        Method for variance estimation:
        - "bootstrap": Block bootstrap at unit level (default)
        - "placebo": Placebo-based variance matching R's synthdid::vcov(method="placebo").
          Implements Algorithm 4 from Arkhangelsky et al. (2021): randomly permutes
          control units, designates N₁ as pseudo-treated, renormalizes original
          weights for remaining pseudo-controls, and computes SDID estimate.
    n_bootstrap : int, default=200
        Number of replications for variance estimation. Used for both:
        - Bootstrap: Number of bootstrap samples
        - Placebo: Number of random permutations (matches R's `replications` argument)
    seed : int, optional
        Random seed for reproducibility. If None (default), results
        will vary between runs.

    Attributes
    ----------
    results_ : SyntheticDiDResults
        Estimation results after calling fit().
    is_fitted_ : bool
        Whether the model has been fitted.

    Examples
    --------
    Basic usage with panel data:

    >>> import pandas as pd
    >>> from diff_diff import SyntheticDiD
    >>>
    >>> # Panel data with units observed over multiple time periods
    >>> # Treatment occurs at period 5 for treated units
    >>> data = pd.DataFrame({
    ...     'unit': [...],      # Unit identifier
    ...     'period': [...],    # Time period
    ...     'outcome': [...],   # Outcome variable
    ...     'treated': [...]    # 1 if unit is ever treated, 0 otherwise
    ... })
    >>>
    >>> # Fit SDID model
    >>> sdid = SyntheticDiD()
    >>> results = sdid.fit(
    ...     data,
    ...     outcome='outcome',
    ...     treatment='treated',
    ...     unit='unit',
    ...     time='period',
    ...     post_periods=[5, 6, 7, 8]
    ... )
    >>>
    >>> # View results
    >>> results.print_summary()
    >>> print(f"ATT: {results.att:.3f} (SE: {results.se:.3f})")
    >>>
    >>> # Examine unit weights
    >>> weights_df = results.get_unit_weights_df()
    >>> print(weights_df.head(10))

    Notes
    -----
    The SDID estimator (Arkhangelsky et al., 2021) computes:

        τ̂ = (Ȳ_treated,post - Σ_t λ_t * Y_treated,t)
            - Σ_j ω_j * (Ȳ_j,post - Σ_t λ_t * Y_j,t)

    Where:
    - ω_j are unit weights (sum to 1, non-negative)
    - λ_t are time weights (sum to 1, non-negative)

    Unit weights ω are chosen to match pre-treatment outcomes:
        min ||Σ_j ω_j * Y_j,pre - Y_treated,pre||²

    This interpolates between:
    - Standard DiD (uniform weights): ω_j = 1/N_control
    - Synthetic Control (exact matching): concentrated weights

    References
    ----------
    Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W., & Wager, S.
    (2021). Synthetic Difference-in-Differences. American Economic Review,
    111(12), 4088-4118.
    """

    def __init__(
        self,
        lambda_reg: float = 0.0,
        zeta: float = 1.0,
        alpha: float = 0.05,
        variance_method: str = "bootstrap",
        n_bootstrap: int = 200,
        seed: Optional[int] = None
    ):
        super().__init__(robust=True, cluster=None, alpha=alpha)
        self.lambda_reg = lambda_reg
        self.zeta = zeta
        self.variance_method = variance_method
        self.n_bootstrap = n_bootstrap
        self.seed = seed

        # Validate variance_method
        valid_methods = ("bootstrap", "placebo")
        if variance_method not in valid_methods:
            raise ValueError(
                f"variance_method must be one of {valid_methods}, "
                f"got '{variance_method}'"
            )

        self._unit_weights = None
        self._time_weights = None

    def fit(  # type: ignore[override]
        self,
        data: pd.DataFrame,
        outcome: str,
        treatment: str,
        unit: str,
        time: str,
        post_periods: Optional[List[Any]] = None,
        covariates: Optional[List[str]] = None
    ) -> SyntheticDiDResults:
        """
        Fit the Synthetic Difference-in-Differences model.

        Parameters
        ----------
        data : pd.DataFrame
            Panel data with observations for multiple units over multiple
            time periods.
        outcome : str
            Name of the outcome variable column.
        treatment : str
            Name of the treatment group indicator column (0/1).
            Should be 1 for all observations of treated units
            (both pre and post treatment).
        unit : str
            Name of the unit identifier column.
        time : str
            Name of the time period column.
        post_periods : list, optional
            List of time period values that are post-treatment.
            If None, uses the last half of periods.
        covariates : list, optional
            List of covariate column names. Covariates are residualized
            out before computing the SDID estimator.

        Returns
        -------
        SyntheticDiDResults
            Object containing the ATT estimate, standard error,
            unit weights, and time weights.

        Raises
        ------
        ValueError
            If required parameters are missing or data validation fails.
        """
        # Validate inputs
        if outcome is None or treatment is None or unit is None or time is None:
            raise ValueError(
                "Must provide 'outcome', 'treatment', 'unit', and 'time'"
            )

        # Check columns exist
        required_cols = [outcome, treatment, unit, time]
        if covariates:
            required_cols.extend(covariates)

        missing = [c for c in required_cols if c not in data.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # Validate treatment is binary
        validate_binary(data[treatment].values, "treatment")

        # Get all unique time periods
        all_periods = sorted(data[time].unique())

        if len(all_periods) < 2:
            raise ValueError("Need at least 2 time periods")

        # Determine pre and post periods
        if post_periods is None:
            mid = len(all_periods) // 2
            post_periods = list(all_periods[mid:])
            pre_periods = list(all_periods[:mid])
        else:
            post_periods = list(post_periods)
            pre_periods = [p for p in all_periods if p not in post_periods]

        if len(post_periods) == 0:
            raise ValueError("Must have at least one post-treatment period")
        if len(pre_periods) == 0:
            raise ValueError("Must have at least one pre-treatment period")

        # Validate post_periods are in data
        for p in post_periods:
            if p not in all_periods:
                raise ValueError(f"Post-period '{p}' not found in time column")

        # Identify treated and control units
        # Treatment indicator should be constant within unit
        unit_treatment = data.groupby(unit)[treatment].first()
        treated_units = unit_treatment[unit_treatment == 1].index.tolist()
        control_units = unit_treatment[unit_treatment == 0].index.tolist()

        if len(treated_units) == 0:
            raise ValueError("No treated units found")
        if len(control_units) == 0:
            raise ValueError("No control units found")

        # Residualize covariates if provided
        working_data = data.copy()
        if covariates:
            working_data = self._residualize_covariates(
                working_data, outcome, covariates, unit, time
            )

        # Create outcome matrices
        # Shape: (n_periods, n_units)
        Y_pre_control, Y_post_control, Y_pre_treated, Y_post_treated = \
            self._create_outcome_matrices(
                working_data, outcome, unit, time,
                pre_periods, post_periods, treated_units, control_units
            )

        # Compute unit weights (synthetic control weights)
        # Average treated outcomes across treated units
        Y_pre_treated_mean = np.mean(Y_pre_treated, axis=1)

        unit_weights = compute_synthetic_weights(
            Y_pre_control,
            Y_pre_treated_mean,
            lambda_reg=self.lambda_reg
        )

        # Compute time weights
        time_weights = compute_time_weights(
            Y_pre_control,
            Y_pre_treated_mean,
            zeta=self.zeta
        )

        # Compute SDID estimate
        Y_post_treated_mean = np.mean(Y_post_treated, axis=1)

        att = compute_sdid_estimator(
            Y_pre_control,
            Y_post_control,
            Y_pre_treated_mean,
            Y_post_treated_mean,
            unit_weights,
            time_weights
        )

        # Compute pre-treatment fit (RMSE)
        synthetic_pre = Y_pre_control @ unit_weights
        pre_fit_rmse = np.sqrt(np.mean((Y_pre_treated_mean - synthetic_pre) ** 2))

        # Compute standard errors based on variance_method
        if self.variance_method == "bootstrap":
            se, bootstrap_estimates = self._bootstrap_se(
                working_data, outcome, unit, time,
                pre_periods, post_periods, treated_units, control_units
            )
            placebo_effects = bootstrap_estimates
            inference_method = "bootstrap"
        else:
            # Use placebo-based variance (R's synthdid Algorithm 4)
            se, placebo_effects = self._placebo_variance_se(
                Y_pre_control,
                Y_post_control,
                Y_pre_treated_mean,
                Y_post_treated_mean,
                unit_weights,
                time_weights,
                n_treated=len(treated_units),
                replications=self.n_bootstrap  # Reuse n_bootstrap for replications
            )
            inference_method = "placebo"

        # Compute test statistics
        if se > 0:
            t_stat = att / se
            # Use placebo distribution for p-value if available
            if len(placebo_effects) > 0:
                # Two-sided p-value from placebo distribution
                p_value = np.mean(np.abs(placebo_effects) >= np.abs(att))
                p_value = max(p_value, 1.0 / (len(placebo_effects) + 1))
            else:
                p_value = compute_p_value(t_stat)
        else:
            t_stat = 0.0
            p_value = 1.0

        # Confidence interval
        conf_int = compute_confidence_interval(att, se, self.alpha)

        # Create weight dictionaries
        unit_weights_dict = {
            unit_id: w for unit_id, w in zip(control_units, unit_weights)
        }
        time_weights_dict = {
            period: w for period, w in zip(pre_periods, time_weights)
        }

        # Store results
        self.results_ = SyntheticDiDResults(
            att=att,
            se=se,
            t_stat=t_stat,
            p_value=p_value,
            conf_int=conf_int,
            n_obs=len(data),
            n_treated=len(treated_units),
            n_control=len(control_units),
            unit_weights=unit_weights_dict,
            time_weights=time_weights_dict,
            pre_periods=pre_periods,
            post_periods=post_periods,
            alpha=self.alpha,
            variance_method=inference_method,
            lambda_reg=self.lambda_reg,
            pre_treatment_fit=pre_fit_rmse,
            placebo_effects=placebo_effects if len(placebo_effects) > 0 else None,
            n_bootstrap=self.n_bootstrap if inference_method == "bootstrap" else None
        )

        self._unit_weights = unit_weights
        self._time_weights = time_weights
        self.is_fitted_ = True

        return self.results_

    def _create_outcome_matrices(
        self,
        data: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        pre_periods: List[Any],
        post_periods: List[Any],
        treated_units: List[Any],
        control_units: List[Any]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create outcome matrices for SDID estimation.

        Returns
        -------
        tuple
            (Y_pre_control, Y_post_control, Y_pre_treated, Y_post_treated)
            Each is a 2D array with shape (n_periods, n_units)
        """
        # Pivot data to wide format
        pivot = data.pivot(index=time, columns=unit, values=outcome)

        # Extract submatrices
        Y_pre_control = pivot.loc[pre_periods, control_units].values
        Y_post_control = pivot.loc[post_periods, control_units].values
        Y_pre_treated = pivot.loc[pre_periods, treated_units].values
        Y_post_treated = pivot.loc[post_periods, treated_units].values

        return (
            Y_pre_control.astype(float),
            Y_post_control.astype(float),
            Y_pre_treated.astype(float),
            Y_post_treated.astype(float)
        )

    def _residualize_covariates(
        self,
        data: pd.DataFrame,
        outcome: str,
        covariates: List[str],
        unit: str,
        time: str
    ) -> pd.DataFrame:
        """
        Residualize outcome by regressing out covariates.

        Uses two-way fixed effects to partial out covariates.
        """
        data = data.copy()

        # Create design matrix with covariates
        X = data[covariates].values.astype(float)

        # Add unit and time dummies
        unit_dummies = pd.get_dummies(data[unit], prefix='u', drop_first=True)
        time_dummies = pd.get_dummies(data[time], prefix='t', drop_first=True)

        X_full = np.column_stack([
            np.ones(len(data)),
            X,
            unit_dummies.values,
            time_dummies.values
        ])

        y = data[outcome].values.astype(float)

        # Fit and get residuals
        coeffs = np.linalg.lstsq(X_full, y, rcond=None)[0]
        residuals = y - X_full @ coeffs

        # Add back the mean for interpretability
        data[outcome] = residuals + np.mean(y)

        return data

    def _bootstrap_se(
        self,
        data: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        pre_periods: List[Any],
        post_periods: List[Any],
        treated_units: List[Any],
        control_units: List[Any]
    ) -> Tuple[float, np.ndarray]:
        """
        Compute bootstrap standard error.

        Uses block bootstrap at the unit level.
        """
        rng = np.random.default_rng(self.seed)

        all_units = treated_units + control_units
        n_units = len(all_units)

        bootstrap_estimates = []

        for _ in range(self.n_bootstrap):
            # Sample units with replacement
            sampled_units = rng.choice(all_units, size=n_units, replace=True)

            # Create bootstrap sample
            boot_data = pd.concat([
                data[data[unit] == u].assign(**{unit: f"{u}_{i}"})
                for i, u in enumerate(sampled_units)
            ], ignore_index=True)

            # Identify treated/control in bootstrap sample
            boot_treated = [
                f"{u}_{i}" for i, u in enumerate(sampled_units)
                if u in treated_units
            ]
            boot_control = [
                f"{u}_{i}" for i, u in enumerate(sampled_units)
                if u in control_units
            ]

            if len(boot_treated) == 0 or len(boot_control) == 0:
                continue

            try:
                # Create matrices
                Y_pre_c, Y_post_c, Y_pre_t, Y_post_t = self._create_outcome_matrices(
                    boot_data, outcome, unit, time,
                    pre_periods, post_periods, boot_treated, boot_control
                )

                # Compute weights
                Y_pre_t_mean = np.mean(Y_pre_t, axis=1)
                Y_post_t_mean = np.mean(Y_post_t, axis=1)

                w = compute_synthetic_weights(Y_pre_c, Y_pre_t_mean, self.lambda_reg)
                t_w = compute_time_weights(Y_pre_c, Y_pre_t_mean, self.zeta)

                # Compute estimate
                tau = compute_sdid_estimator(
                    Y_pre_c, Y_post_c, Y_pre_t_mean, Y_post_t_mean, w, t_w
                )
                bootstrap_estimates.append(tau)

            except (ValueError, LinAlgError, KeyError):
                # Skip failed bootstrap iterations (e.g., singular matrices,
                # missing data in resampled units, or invalid weight computations)
                continue

        bootstrap_estimates = np.array(bootstrap_estimates)

        # Check bootstrap success rate and handle failures appropriately
        n_successful = len(bootstrap_estimates)
        failure_rate = 1 - (n_successful / self.n_bootstrap)

        if n_successful == 0:
            raise ValueError(
                f"All {self.n_bootstrap} bootstrap iterations failed. "
                f"This typically occurs when:\n"
                f"  - Sample size is too small for reliable resampling\n"
                f"  - Weight matrices are singular or near-singular\n"
                f"  - Insufficient pre-treatment periods for weight estimation\n"
                f"  - Too few control units relative to treated units\n"
                f"Consider using n_bootstrap=0 to disable bootstrap inference "
                f"and rely on placebo-based standard errors, or increase "
                f"the regularization parameters (lambda_reg, zeta)."
            )
        elif n_successful == 1:
            warnings.warn(
                f"Only 1/{self.n_bootstrap} bootstrap iteration succeeded. "
                f"Standard error cannot be computed reliably (requires at least 2). "
                f"Returning SE=0.0. Consider the suggestions above for improving "
                f"bootstrap convergence.",
                UserWarning,
                stacklevel=2,
            )
            se = 0.0
        elif failure_rate > 0.05:
            warnings.warn(
                f"Only {n_successful}/{self.n_bootstrap} bootstrap iterations succeeded "
                f"({failure_rate:.1%} failure rate). Standard errors may be unreliable. "
                f"This can occur with small samples, near-singular weight matrices, "
                f"or insufficient pre-treatment periods.",
                UserWarning,
                stacklevel=2,
            )
            se = np.std(bootstrap_estimates, ddof=1)
        else:
            se = np.std(bootstrap_estimates, ddof=1)

        return se, bootstrap_estimates

    def _placebo_variance_se(
        self,
        Y_pre_control: np.ndarray,
        Y_post_control: np.ndarray,
        Y_pre_treated_mean: np.ndarray,
        Y_post_treated_mean: np.ndarray,
        unit_weights: np.ndarray,
        time_weights: np.ndarray,
        n_treated: int,
        replications: int = 200
    ) -> Tuple[float, np.ndarray]:
        """
        Compute placebo-based variance matching R's synthdid methodology.

        This implements Algorithm 4 from Arkhangelsky et al. (2021),
        matching R's synthdid::vcov(method = "placebo"):

        1. Randomly sample N₀ control indices (permutation)
        2. Designate last N₁ as pseudo-treated, first (N₀-N₁) as pseudo-controls
        3. Renormalize original unit weights for pseudo-controls
        4. Compute SDID estimate using renormalized weights
        5. Repeat `replications` times
        6. SE = sqrt((r-1)/r) * sd(estimates)

        Parameters
        ----------
        Y_pre_control : np.ndarray
            Control outcomes in pre-treatment periods, shape (n_pre, n_control).
        Y_post_control : np.ndarray
            Control outcomes in post-treatment periods, shape (n_post, n_control).
        Y_pre_treated_mean : np.ndarray
            Mean treated outcomes in pre-treatment periods, shape (n_pre,).
        Y_post_treated_mean : np.ndarray
            Mean treated outcomes in post-treatment periods, shape (n_post,).
        unit_weights : np.ndarray
            Original unit weights from main estimation, shape (n_control,).
        time_weights : np.ndarray
            Time weights from main estimation, shape (n_pre,).
        n_treated : int
            Number of treated units in the original estimation.
        replications : int, default=200
            Number of placebo replications.

        Returns
        -------
        tuple
            (se, placebo_effects) where se is the standard error and
            placebo_effects is the array of placebo treatment effects.

        References
        ----------
        Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W., & Wager, S.
        (2021). Synthetic Difference-in-Differences. American Economic Review,
        111(12), 4088-4118. Algorithm 4.
        """
        rng = np.random.default_rng(self.seed)
        n_pre, n_control = Y_pre_control.shape

        # Ensure we have enough controls for the split
        n_pseudo_control = n_control - n_treated
        if n_pseudo_control < 1:
            warnings.warn(
                f"Not enough control units ({n_control}) for placebo variance "
                f"estimation with {n_treated} treated units. "
                f"Consider using variance_method='bootstrap'.",
                UserWarning,
                stacklevel=3,
            )
            return 0.0, np.array([])

        placebo_estimates = []

        for _ in range(replications):
            try:
                # Random permutation of control indices (Algorithm 4, step 1)
                perm = rng.permutation(n_control)

                # Split into pseudo-controls and pseudo-treated (step 2)
                pseudo_control_idx = perm[:n_pseudo_control]
                pseudo_treated_idx = perm[n_pseudo_control:]

                # Renormalize original weights for pseudo-controls (step 3)
                # This keeps the relative importance from the main estimation
                pseudo_weights = unit_weights[pseudo_control_idx]
                weight_sum = pseudo_weights.sum()
                if weight_sum > 0:
                    pseudo_weights = pseudo_weights / weight_sum
                else:
                    # Fallback to uniform if weights sum to zero
                    pseudo_weights = np.ones(n_pseudo_control) / n_pseudo_control

                # Get pseudo-treated outcomes (mean across pseudo-treated units)
                Y_pre_pseudo_treated = np.mean(
                    Y_pre_control[:, pseudo_treated_idx], axis=1
                )
                Y_post_pseudo_treated = np.mean(
                    Y_post_control[:, pseudo_treated_idx], axis=1
                )

                # Get pseudo-control outcomes
                Y_pre_pseudo_control = Y_pre_control[:, pseudo_control_idx]
                Y_post_pseudo_control = Y_post_control[:, pseudo_control_idx]

                # Compute placebo SDID estimate (step 4)
                tau = compute_sdid_estimator(
                    Y_pre_pseudo_control,
                    Y_post_pseudo_control,
                    Y_pre_pseudo_treated,
                    Y_post_pseudo_treated,
                    pseudo_weights,
                    time_weights
                )
                placebo_estimates.append(tau)

            except (ValueError, LinAlgError, ZeroDivisionError):
                # Skip failed iterations
                continue

        placebo_estimates = np.array(placebo_estimates)
        n_successful = len(placebo_estimates)

        if n_successful < 2:
            warnings.warn(
                f"Only {n_successful} placebo replications completed successfully. "
                f"Standard error cannot be estimated reliably. "
                f"Consider using variance_method='bootstrap' or increasing "
                f"the number of control units.",
                UserWarning,
                stacklevel=3,
            )
            return 0.0, placebo_estimates

        # Warn if many replications failed
        failure_rate = 1 - (n_successful / replications)
        if failure_rate > 0.05:
            warnings.warn(
                f"Only {n_successful}/{replications} placebo replications succeeded "
                f"({failure_rate:.1%} failure rate). Standard errors may be unreliable.",
                UserWarning,
                stacklevel=3,
            )

        # Compute SE using R's formula: sqrt((r-1)/r) * sd(estimates)
        # This matches synthdid::vcov.R exactly
        se = np.sqrt((n_successful - 1) / n_successful) * np.std(
            placebo_estimates, ddof=1
        )

        return se, placebo_estimates

    def get_params(self) -> Dict[str, Any]:
        """Get estimator parameters."""
        return {
            "lambda_reg": self.lambda_reg,
            "zeta": self.zeta,
            "alpha": self.alpha,
            "variance_method": self.variance_method,
            "n_bootstrap": self.n_bootstrap,
            "seed": self.seed,
        }

    def set_params(self, **params) -> "SyntheticDiD":
        """Set estimator parameters."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")
        return self
