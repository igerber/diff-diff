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
        - "placebo": Placebo-based variance using jackknife formula,
          matching R's synthdid package methodology (Arkhangelsky et al. 2021)
    n_bootstrap : int, default=200
        Number of bootstrap replications when variance_method="bootstrap".
        Ignored when variance_method="placebo".
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
            # Use placebo-based variance (R's synthdid methodology)
            se, placebo_effects = self._placebo_variance_se(
                Y_pre_control,
                Y_post_control,
                Y_pre_treated_mean,
                time_weights,
                control_units
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
        time_weights: np.ndarray,
        control_units: List[Any]
    ) -> Tuple[float, np.ndarray]:
        """
        Compute placebo-based variance using R's synthdid methodology.

        This implements the jackknife-style variance estimator from
        Arkhangelsky et al. (2021), matching R's synthdid package.

        For each control unit i, we pretend it was treated and compute
        the SDID estimate using the remaining controls. The variance is:

            Var(tau) = ((N0 - 1) / N0) * sum((tau_i - tau_bar)^2)

        Parameters
        ----------
        Y_pre_control : np.ndarray
            Control outcomes in pre-treatment periods, shape (n_pre, n_control).
        Y_post_control : np.ndarray
            Control outcomes in post-treatment periods, shape (n_post, n_control).
        Y_pre_treated_mean : np.ndarray
            Mean treated outcomes in pre-treatment periods, shape (n_pre,).
        time_weights : np.ndarray
            Time weights from main estimation, shape (n_pre,).
        control_units : list
            List of control unit identifiers.

        Returns
        -------
        tuple
            (se, placebo_effects) where se is the standard error and
            placebo_effects is the array of placebo treatment effects.
        """
        n_pre, n_control = Y_pre_control.shape

        placebo_effects = []

        for j in range(n_control):
            # Treat unit j as the "treated" unit
            Y_pre_placebo_treated = Y_pre_control[:, j]
            Y_post_placebo_treated = Y_post_control[:, j]

            # Use remaining units as controls
            remaining_idx = [i for i in range(n_control) if i != j]

            if len(remaining_idx) == 0:
                continue

            Y_pre_remaining = Y_pre_control[:, remaining_idx]
            Y_post_remaining = Y_post_control[:, remaining_idx]

            try:
                # Recompute weights for remaining controls
                remaining_weights = compute_synthetic_weights(
                    Y_pre_remaining,
                    Y_pre_placebo_treated,
                    lambda_reg=self.lambda_reg
                )

                # Compute placebo effect using the SDID formula
                placebo_tau = compute_sdid_estimator(
                    Y_pre_remaining,
                    Y_post_remaining,
                    Y_pre_placebo_treated,
                    Y_post_placebo_treated,
                    remaining_weights,
                    time_weights
                )

                placebo_effects.append(placebo_tau)

            except (ValueError, LinAlgError):
                # Skip failed placebo computations
                continue

        placebo_effects = np.array(placebo_effects)

        if len(placebo_effects) < 2:
            warnings.warn(
                f"Only {len(placebo_effects)} placebo effects computed successfully. "
                f"Standard error cannot be estimated reliably. "
                f"Consider using variance_method='bootstrap' or increasing "
                f"the number of control units.",
                UserWarning,
                stacklevel=3,
            )
            return 0.0, placebo_effects

        # Compute variance using R's synthdid jackknife formula:
        # Var = ((N0 - 1) / N0) * sum((tau_i - tau_bar)^2)
        n_placebo = len(placebo_effects)
        placebo_mean = np.mean(placebo_effects)
        variance = ((n_placebo - 1) / n_placebo) * np.sum(
            (placebo_effects - placebo_mean) ** 2
        )
        se = np.sqrt(variance)

        return se, placebo_effects

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
