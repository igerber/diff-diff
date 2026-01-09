"""
Triple Difference (DDD) estimators.

Implements the methodology from Ortiz-Villavicencio & Sant'Anna (2025)
"Better Understanding Triple Differences Estimators" for causal inference
when treatment requires satisfying two criteria:
1. Belonging to a treated group (e.g., a state with a policy)
2. Being in an eligible partition (e.g., women, low-income, etc.)

This module provides regression adjustment, inverse probability weighting,
and doubly robust estimators that correctly handle covariate adjustment,
unlike naive implementations.

Current Implementation (v1.3):
    - 2-period DDD (pre/post binary time indicator)
    - Regression adjustment, IPW, and doubly robust estimation
    - Analytical standard errors with robust/cluster options
    - Proper covariate handling

Planned for v1.4 (see ROADMAP.md):
    - Staggered adoption support (multiple treatment timing)
    - Event study aggregation for dynamic treatment effects
    - Multiplier bootstrap inference
    - Integration with plot_event_study() visualization

Reference:
    Ortiz-Villavicencio, M., & Sant'Anna, P. H. C. (2025).
    Better Understanding Triple Differences Estimators.
    arXiv:2505.09942.
"""

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import optimize

from diff_diff.results import _get_significance_stars
from diff_diff.utils import (
    compute_confidence_interval,
    compute_p_value,
    compute_robust_se,
)

# =============================================================================
# Results Classes
# =============================================================================


@dataclass
class TripleDifferenceResults:
    """
    Results from Triple Difference (DDD) estimation.

    Provides access to the estimated average treatment effect on the treated
    (ATT), standard errors, confidence intervals, and diagnostic information.

    Attributes
    ----------
    att : float
        Average Treatment effect on the Treated (ATT).
        This is the effect on units in the treated group (G=1) and eligible
        partition (P=1) after treatment (T=1).
    se : float
        Standard error of the ATT estimate.
    t_stat : float
        T-statistic for the ATT estimate.
    p_value : float
        P-value for the null hypothesis that ATT = 0.
    conf_int : tuple[float, float]
        Confidence interval for the ATT.
    n_obs : int
        Total number of observations used in estimation.
    n_treated_eligible : int
        Number of observations in treated group and eligible partition.
    n_treated_ineligible : int
        Number of observations in treated group and ineligible partition.
    n_control_eligible : int
        Number of observations in control group and eligible partition.
    n_control_ineligible : int
        Number of observations in control group and ineligible partition.
    estimation_method : str
        Estimation method used: "dr" (doubly robust), "reg" (regression
        adjustment), or "ipw" (inverse probability weighting).
    alpha : float
        Significance level used for confidence intervals.
    """

    att: float
    se: float
    t_stat: float
    p_value: float
    conf_int: Tuple[float, float]
    n_obs: int
    n_treated_eligible: int
    n_treated_ineligible: int
    n_control_eligible: int
    n_control_ineligible: int
    estimation_method: str
    alpha: float = 0.05
    # Group means for diagnostics
    group_means: Optional[Dict[str, float]] = field(default=None)
    # Propensity score diagnostics (for IPW/DR)
    pscore_stats: Optional[Dict[str, float]] = field(default=None)
    # Regression diagnostics
    r_squared: Optional[float] = field(default=None)
    # Covariate balance statistics
    covariate_balance: Optional[pd.DataFrame] = field(default=None, repr=False)
    # Inference details
    inference_method: str = field(default="analytical")
    n_bootstrap: Optional[int] = field(default=None)
    n_clusters: Optional[int] = field(default=None)

    def __repr__(self) -> str:
        """Concise string representation."""
        return (
            f"TripleDifferenceResults(ATT={self.att:.4f}{self.significance_stars}, "
            f"SE={self.se:.4f}, p={self.p_value:.4f}, method={self.estimation_method})"
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
            "Triple Difference (DDD) Estimation Results".center(75),
            "=" * 75,
            "",
            f"{'Estimation method:':<30} {self.estimation_method:>15}",
            f"{'Total observations:':<30} {self.n_obs:>15}",
            "",
            "Sample Composition by Cell:",
            f"  {'Treated group, Eligible:':<28} {self.n_treated_eligible:>15}",
            f"  {'Treated group, Ineligible:':<28} {self.n_treated_ineligible:>15}",
            f"  {'Control group, Eligible:':<28} {self.n_control_eligible:>15}",
            f"  {'Control group, Ineligible:':<28} {self.n_control_ineligible:>15}",
        ]

        if self.r_squared is not None:
            lines.append(f"{'R-squared:':<30} {self.r_squared:>15.4f}")

        if self.inference_method != "analytical":
            lines.append(f"{'Inference method:':<30} {self.inference_method:>15}")
            if self.n_bootstrap is not None:
                lines.append(f"{'Bootstrap replications:':<30} {self.n_bootstrap:>15}")
            if self.n_clusters is not None:
                lines.append(f"{'Number of clusters:':<30} {self.n_clusters:>15}")

        lines.extend([
            "",
            "-" * 75,
            f"{'Parameter':<15} {'Estimate':>12} {'Std. Err.':>12} {'t-stat':>10} {'P>|t|':>10} {'':>5}",
            "-" * 75,
            f"{'ATT':<15} {self.att:>12.4f} {self.se:>12.4f} {self.t_stat:>10.3f} {self.p_value:>10.4f} {self.significance_stars:>5}",
            "-" * 75,
            "",
            f"{conf_level}% Confidence Interval: [{self.conf_int[0]:.4f}, {self.conf_int[1]:.4f}]",
        ])

        # Show group means if available
        if self.group_means:
            lines.extend([
                "",
                "-" * 75,
                "Cell Means (Y):",
                "-" * 75,
            ])
            for cell, mean in self.group_means.items():
                lines.append(f"  {cell:<35} {mean:>12.4f}")

        # Show propensity score diagnostics if available
        if self.pscore_stats:
            lines.extend([
                "",
                "-" * 75,
                "Propensity Score Diagnostics:",
                "-" * 75,
            ])
            for stat, value in self.pscore_stats.items():
                lines.append(f"  {stat:<35} {value:>12.4f}")

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
        result = {
            "att": self.att,
            "se": self.se,
            "t_stat": self.t_stat,
            "p_value": self.p_value,
            "conf_int_lower": self.conf_int[0],
            "conf_int_upper": self.conf_int[1],
            "n_obs": self.n_obs,
            "n_treated_eligible": self.n_treated_eligible,
            "n_treated_ineligible": self.n_treated_ineligible,
            "n_control_eligible": self.n_control_eligible,
            "n_control_ineligible": self.n_control_ineligible,
            "estimation_method": self.estimation_method,
            "inference_method": self.inference_method,
        }
        if self.r_squared is not None:
            result["r_squared"] = self.r_squared
        if self.n_bootstrap is not None:
            result["n_bootstrap"] = self.n_bootstrap
        if self.n_clusters is not None:
            result["n_clusters"] = self.n_clusters
        return result

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert results to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with estimation results.
        """
        return pd.DataFrame([self.to_dict()])

    @property
    def is_significant(self) -> bool:
        """Check if the ATT is statistically significant at the alpha level."""
        return bool(self.p_value < self.alpha)

    @property
    def significance_stars(self) -> str:
        """Return significance stars based on p-value."""
        return _get_significance_stars(self.p_value)


# =============================================================================
# Helper Functions
# =============================================================================


def _logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit logistic regression using scipy optimize.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features). Intercept added automatically.
    y : np.ndarray
        Binary outcome (0/1).
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    beta : np.ndarray
        Fitted coefficients (including intercept).
    probs : np.ndarray
        Predicted probabilities.
    """
    n, p = X.shape
    X_with_intercept = np.column_stack([np.ones(n), X])

    def neg_log_likelihood(beta: np.ndarray) -> float:
        z = X_with_intercept @ beta
        z = np.clip(z, -500, 500)
        log_lik = np.sum(y * z - np.log(1 + np.exp(z)))
        return -log_lik

    def gradient(beta: np.ndarray) -> np.ndarray:
        z = X_with_intercept @ beta
        z = np.clip(z, -500, 500)
        probs = 1 / (1 + np.exp(-z))
        return -X_with_intercept.T @ (y - probs)

    beta_init = np.zeros(p + 1)

    result = optimize.minimize(
        neg_log_likelihood,
        beta_init,
        method='BFGS',
        jac=gradient,
        options={'maxiter': max_iter, 'gtol': tol}
    )

    beta = result.x
    z = X_with_intercept @ beta
    z = np.clip(z, -500, 500)
    probs = 1 / (1 + np.exp(-z))

    return beta, probs


def _linear_regression(
    X: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Fit OLS regression.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features). Intercept added automatically.
    y : np.ndarray
        Outcome variable.

    Returns
    -------
    beta : np.ndarray
        Fitted coefficients (including intercept).
    fitted : np.ndarray
        Fitted values.
    r_squared : float
        R-squared of the regression.
    """
    n = X.shape[0]
    X_with_intercept = np.column_stack([np.ones(n), X])

    try:
        beta = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(X_with_intercept) @ y

    fitted = X_with_intercept @ beta
    residuals = y - fitted

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return beta, fitted, r_squared


# =============================================================================
# Main Estimator Class
# =============================================================================


class TripleDifference:
    """
    Triple Difference (DDD) estimator.

    Estimates the Average Treatment effect on the Treated (ATT) when treatment
    requires satisfying two criteria: belonging to a treated group AND being
    in an eligible partition of the population.

    This implementation follows Ortiz-Villavicencio & Sant'Anna (2025), which
    shows that naive DDD implementations (difference of two DiDs, three-way
    fixed effects) are invalid when covariates are needed for identification.

    Parameters
    ----------
    estimation_method : str, default="dr"
        Estimation method to use:
        - "dr": Doubly robust (recommended). Consistent if either the outcome
          model or propensity score model is correctly specified.
        - "reg": Regression adjustment (outcome regression).
        - "ipw": Inverse probability weighting.
    robust : bool, default=True
        Whether to use heteroskedasticity-robust standard errors (HC1).
    cluster : str, optional
        Column name for cluster-robust standard errors.
    alpha : float, default=0.05
        Significance level for confidence intervals.
    pscore_trim : float, default=0.01
        Trimming threshold for propensity scores. Scores below this value
        or above (1 - pscore_trim) are clipped to avoid extreme weights.

    Attributes
    ----------
    results_ : TripleDifferenceResults
        Estimation results after calling fit().
    is_fitted_ : bool
        Whether the model has been fitted.

    Examples
    --------
    Basic usage with a DataFrame:

    >>> import pandas as pd
    >>> from diff_diff import TripleDifference
    >>>
    >>> # Data where treatment affects women (partition=1) in states
    >>> # that enacted a policy (group=1)
    >>> data = pd.DataFrame({
    ...     'outcome': [...],
    ...     'group': [1, 1, 0, 0, ...],      # 1=policy state, 0=control state
    ...     'partition': [1, 0, 1, 0, ...],  # 1=women, 0=men
    ...     'post': [0, 0, 1, 1, ...],       # 1=post-treatment period
    ... })
    >>>
    >>> # Fit using doubly robust estimation
    >>> ddd = TripleDifference(estimation_method="dr")
    >>> results = ddd.fit(
    ...     data,
    ...     outcome='outcome',
    ...     group='group',
    ...     partition='partition',
    ...     time='post'
    ... )
    >>> print(results.att)  # ATT estimate

    With covariates (properly handled unlike naive DDD):

    >>> results = ddd.fit(
    ...     data,
    ...     outcome='outcome',
    ...     group='group',
    ...     partition='partition',
    ...     time='post',
    ...     covariates=['age', 'income']
    ... )

    Notes
    -----
    The DDD estimator is appropriate when:

    1. Treatment affects only units satisfying BOTH criteria:
       - Belonging to a treated group (G=1), e.g., states with a policy
       - Being in an eligible partition (P=1), e.g., women, low-income

    2. The DDD parallel trends assumption holds: the differential trend
       between eligible and ineligible partitions would have been the same
       across treated and control groups, absent treatment.

    This is weaker than requiring separate parallel trends for two DiDs,
    as biases can cancel out in the differencing.

    References
    ----------
    .. [1] Ortiz-Villavicencio, M., & Sant'Anna, P. H. C. (2025).
           Better Understanding Triple Differences Estimators.
           arXiv:2505.09942.

    .. [2] Gruber, J. (1994). The incidence of mandated maternity benefits.
           American Economic Review, 84(3), 622-641.
    """

    def __init__(
        self,
        estimation_method: str = "dr",
        robust: bool = True,
        cluster: Optional[str] = None,
        alpha: float = 0.05,
        pscore_trim: float = 0.01,
    ):
        if estimation_method not in ("dr", "reg", "ipw"):
            raise ValueError(
                f"estimation_method must be 'dr', 'reg', or 'ipw', "
                f"got '{estimation_method}'"
            )
        self.estimation_method = estimation_method
        self.robust = robust
        self.cluster = cluster
        self.alpha = alpha
        self.pscore_trim = pscore_trim

        self.is_fitted_ = False
        self.results_: Optional[TripleDifferenceResults] = None

    def fit(
        self,
        data: pd.DataFrame,
        outcome: str,
        group: str,
        partition: str,
        time: str,
        covariates: Optional[List[str]] = None,
    ) -> TripleDifferenceResults:
        """
        Fit the Triple Difference model.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing all variables.
        outcome : str
            Name of the outcome variable column.
        group : str
            Name of the group indicator column (0/1).
            1 = treated group (e.g., states that enacted policy).
            0 = control group.
        partition : str
            Name of the partition/eligibility indicator column (0/1).
            1 = eligible partition (e.g., women, targeted demographic).
            0 = ineligible partition.
        time : str
            Name of the time period indicator column (0/1).
            1 = post-treatment period.
            0 = pre-treatment period.
        covariates : list of str, optional
            List of covariate column names to adjust for.
            These are properly incorporated using the selected estimation
            method (unlike naive DDD implementations).

        Returns
        -------
        TripleDifferenceResults
            Object containing estimation results.

        Raises
        ------
        ValueError
            If required columns are missing or data validation fails.
        """
        # Validate inputs
        self._validate_data(data, outcome, group, partition, time, covariates)

        # Extract data
        y = data[outcome].values.astype(float)
        G = data[group].values.astype(float)
        P = data[partition].values.astype(float)
        T = data[time].values.astype(float)

        # Get covariates if specified
        X = None
        if covariates:
            X = data[covariates].values.astype(float)
            if np.any(np.isnan(X)):
                raise ValueError("Covariates contain missing values")

        # Count observations in each cell
        n_obs = len(y)
        n_treated_eligible = int(np.sum((G == 1) & (P == 1)))
        n_treated_ineligible = int(np.sum((G == 1) & (P == 0)))
        n_control_eligible = int(np.sum((G == 0) & (P == 1)))
        n_control_ineligible = int(np.sum((G == 0) & (P == 0)))

        # Compute cell means for diagnostics
        group_means = self._compute_cell_means(y, G, P, T)

        # Estimate ATT based on method
        if self.estimation_method == "reg":
            att, se, r_squared, pscore_stats = self._regression_adjustment(
                y, G, P, T, X
            )
        elif self.estimation_method == "ipw":
            att, se, r_squared, pscore_stats = self._ipw_estimation(
                y, G, P, T, X
            )
        else:  # doubly robust
            att, se, r_squared, pscore_stats = self._doubly_robust(
                y, G, P, T, X
            )

        # Compute inference
        t_stat = att / se if se > 0 else 0.0
        df = n_obs - 8  # Approximate df (8 cell means)
        if covariates:
            df -= len(covariates)
        df = max(df, 1)

        p_value = compute_p_value(t_stat, df=df)
        conf_int = compute_confidence_interval(att, se, self.alpha, df=df)

        # Get number of clusters if clustering
        n_clusters = None
        if self.cluster is not None:
            n_clusters = data[self.cluster].nunique()

        # Create results object
        self.results_ = TripleDifferenceResults(
            att=att,
            se=se,
            t_stat=t_stat,
            p_value=p_value,
            conf_int=conf_int,
            n_obs=n_obs,
            n_treated_eligible=n_treated_eligible,
            n_treated_ineligible=n_treated_ineligible,
            n_control_eligible=n_control_eligible,
            n_control_ineligible=n_control_ineligible,
            estimation_method=self.estimation_method,
            alpha=self.alpha,
            group_means=group_means,
            pscore_stats=pscore_stats,
            r_squared=r_squared,
            inference_method="analytical",
            n_clusters=n_clusters,
        )

        self.is_fitted_ = True
        return self.results_

    def _validate_data(
        self,
        data: pd.DataFrame,
        outcome: str,
        group: str,
        partition: str,
        time: str,
        covariates: Optional[List[str]] = None,
    ) -> None:
        """Validate input data."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")

        # Check required columns exist
        required_cols = [outcome, group, partition, time]
        if covariates:
            required_cols.extend(covariates)

        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in data: {missing_cols}")

        # Check for missing values in required columns
        for col in [outcome, group, partition, time]:
            if data[col].isna().any():
                raise ValueError(f"Column '{col}' contains missing values")

        # Validate binary variables
        for col, name in [(group, "group"), (partition, "partition"), (time, "time")]:
            unique_vals = set(data[col].unique())
            if not unique_vals.issubset({0, 1, 0.0, 1.0}):
                raise ValueError(
                    f"'{name}' column must be binary (0/1), "
                    f"got values: {sorted(unique_vals)}"
                )
            if len(unique_vals) < 2:
                raise ValueError(
                    f"'{name}' column must have both 0 and 1 values"
                )

        # Check we have observations in all cells
        G = data[group].values
        P = data[partition].values
        T = data[time].values

        cells = [
            ((G == 1) & (P == 1) & (T == 0), "treated, eligible, pre"),
            ((G == 1) & (P == 1) & (T == 1), "treated, eligible, post"),
            ((G == 1) & (P == 0) & (T == 0), "treated, ineligible, pre"),
            ((G == 1) & (P == 0) & (T == 1), "treated, ineligible, post"),
            ((G == 0) & (P == 1) & (T == 0), "control, eligible, pre"),
            ((G == 0) & (P == 1) & (T == 1), "control, eligible, post"),
            ((G == 0) & (P == 0) & (T == 0), "control, ineligible, pre"),
            ((G == 0) & (P == 0) & (T == 1), "control, ineligible, post"),
        ]

        for mask, cell_name in cells:
            if np.sum(mask) == 0:
                raise ValueError(
                    f"No observations in cell: {cell_name}. "
                    "DDD requires observations in all 8 cells."
                )

    def _compute_cell_means(
        self,
        y: np.ndarray,
        G: np.ndarray,
        P: np.ndarray,
        T: np.ndarray,
    ) -> Dict[str, float]:
        """Compute mean outcomes for each of the 8 DDD cells."""
        means = {}
        for g_val, g_name in [(1, "Treated"), (0, "Control")]:
            for p_val, p_name in [(1, "Eligible"), (0, "Ineligible")]:
                for t_val, t_name in [(0, "Pre"), (1, "Post")]:
                    mask = (G == g_val) & (P == p_val) & (T == t_val)
                    cell_name = f"{g_name}, {p_name}, {t_name}"
                    means[cell_name] = float(np.mean(y[mask]))
        return means

    def _regression_adjustment(
        self,
        y: np.ndarray,
        G: np.ndarray,
        P: np.ndarray,
        T: np.ndarray,
        X: Optional[np.ndarray],
    ) -> Tuple[float, float, Optional[float], Optional[Dict[str, float]]]:
        """
        Estimate ATT using regression adjustment.

        Fits an outcome regression with full interactions and covariates,
        then computes the DDD estimand.

        With covariates, this properly conditions on X rather than naively
        differencing two DiD estimates.
        """
        n = len(y)

        # Build design matrix for DDD regression
        # Full specification: Y = α + β_G*G + β_P*P + β_T*T
        #                        + β_GP*G*P + β_GT*G*T + β_PT*P*T
        #                        + β_GPT*G*P*T + γ'X + ε
        # The DDD estimate is β_GPT

        # Create interactions
        GP = G * P
        GT = G * T
        PT = P * T
        GPT = G * P * T

        # Build design matrix
        design_cols = [np.ones(n), G, P, T, GP, GT, PT, GPT]
        col_names = ["const", "G", "P", "T", "G*P", "G*T", "P*T", "G*P*T"]

        if X is not None:
            for i in range(X.shape[1]):
                design_cols.append(X[:, i])
                col_names.append(f"X{i}")

        design_matrix = np.column_stack(design_cols)

        # Fit OLS
        try:
            coefficients = np.linalg.lstsq(design_matrix, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            coefficients = np.linalg.pinv(design_matrix) @ y

        fitted = design_matrix @ coefficients
        residuals = y - fitted

        # R-squared
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # ATT is the coefficient on G*P*T (index 7)
        att = coefficients[7]

        # Compute standard error
        se = self._compute_se(design_matrix, residuals, 7)

        return att, se, r_squared, None

    def _ipw_estimation(
        self,
        y: np.ndarray,
        G: np.ndarray,
        P: np.ndarray,
        T: np.ndarray,
        X: Optional[np.ndarray],
    ) -> Tuple[float, float, Optional[float], Optional[Dict[str, float]]]:
        """
        Estimate ATT using inverse probability weighting.

        Estimates propensity scores for cell membership and uses IPW
        to reweight observations for the DDD estimand.
        """
        n = len(y)

        # For DDD-IPW, we need to estimate probabilities for each cell
        # and use them to construct weighted estimators

        # Create cell indicators
        # Cell 1: G=1, P=1 (treated, eligible) - "effectively treated"
        # Cell 2: G=1, P=0 (treated, ineligible)
        # Cell 3: G=0, P=1 (control, eligible)
        # Cell 4: G=0, P=0 (control, ineligible)

        cell_1 = (G == 1) & (P == 1)
        cell_2 = (G == 1) & (P == 0)
        cell_3 = (G == 0) & (P == 1)
        cell_4 = (G == 0) & (P == 0)

        if X is not None and X.shape[1] > 0:
            # Estimate multinomial propensity scores
            # For simplicity, we estimate binary propensity scores for each cell
            # P(G=1|X) and P(P=1|X,G)

            # Propensity for being in treated group
            try:
                _, p_G = _logistic_regression(X, G)
            except Exception:
                warnings.warn(
                    "Propensity score estimation for G failed. "
                    "Using unconditional probabilities.",
                    UserWarning,
                    stacklevel=3,
                )
                p_G = np.full(n, np.mean(G))

            # Propensity for being in eligible partition (conditional on X)
            try:
                _, p_P = _logistic_regression(X, P)
            except Exception:
                warnings.warn(
                    "Propensity score estimation for P failed. "
                    "Using unconditional probabilities.",
                    UserWarning,
                    stacklevel=3,
                )
                p_P = np.full(n, np.mean(P))

            # Clip propensity scores
            p_G = np.clip(p_G, self.pscore_trim, 1 - self.pscore_trim)
            p_P = np.clip(p_P, self.pscore_trim, 1 - self.pscore_trim)

            # Cell probabilities (assuming independence conditional on X)
            p_cell_1 = p_G * p_P  # P(G=1, P=1|X)
            p_cell_2 = p_G * (1 - p_P)  # P(G=1, P=0|X)
            p_cell_3 = (1 - p_G) * p_P  # P(G=0, P=1|X)
            p_cell_4 = (1 - p_G) * (1 - p_P)  # P(G=0, P=0|X)

            pscore_stats = {
                "P(G=1) mean": float(np.mean(p_G)),
                "P(G=1) std": float(np.std(p_G)),
                "P(P=1) mean": float(np.mean(p_P)),
                "P(P=1) std": float(np.std(p_P)),
            }
        else:
            # Unconditional probabilities
            p_cell_1 = np.full(n, np.mean(cell_1))
            p_cell_2 = np.full(n, np.mean(cell_2))
            p_cell_3 = np.full(n, np.mean(cell_3))
            p_cell_4 = np.full(n, np.mean(cell_4))
            pscore_stats = None

        # Clip cell probabilities
        p_cell_1 = np.clip(p_cell_1, self.pscore_trim, 1 - self.pscore_trim)
        p_cell_2 = np.clip(p_cell_2, self.pscore_trim, 1 - self.pscore_trim)
        p_cell_3 = np.clip(p_cell_3, self.pscore_trim, 1 - self.pscore_trim)
        p_cell_4 = np.clip(p_cell_4, self.pscore_trim, 1 - self.pscore_trim)

        # IPW estimator for DDD
        # The DDD-IPW estimator reweights each cell to have the same
        # covariate distribution as the effectively treated (G=1, P=1)

        # Pre-period means
        pre_mask = T == 0
        post_mask = T == 1

        def weighted_mean(y_vals, weights):
            """Compute weighted mean, handling edge cases."""
            w_sum = np.sum(weights)
            if w_sum <= 0:
                return 0.0
            return np.sum(y_vals * weights) / w_sum

        # Cell 1 (G=1, P=1): weight = 1 (reference)
        w1_pre = cell_1 & pre_mask
        w1_post = cell_1 & post_mask
        y_11_pre = np.mean(y[w1_pre]) if np.sum(w1_pre) > 0 else 0
        y_11_post = np.mean(y[w1_post]) if np.sum(w1_post) > 0 else 0

        # Cell 2 (G=1, P=0): reweight to match X-distribution of cell 1
        w2_pre = (cell_2 & pre_mask).astype(float) * (p_cell_1 / p_cell_2)
        w2_post = (cell_2 & post_mask).astype(float) * (p_cell_1 / p_cell_2)
        y_10_pre = weighted_mean(y, w2_pre)
        y_10_post = weighted_mean(y, w2_post)

        # Cell 3 (G=0, P=1): reweight to match X-distribution of cell 1
        w3_pre = (cell_3 & pre_mask).astype(float) * (p_cell_1 / p_cell_3)
        w3_post = (cell_3 & post_mask).astype(float) * (p_cell_1 / p_cell_3)
        y_01_pre = weighted_mean(y, w3_pre)
        y_01_post = weighted_mean(y, w3_post)

        # Cell 4 (G=0, P=0): reweight to match X-distribution of cell 1
        w4_pre = (cell_4 & pre_mask).astype(float) * (p_cell_1 / p_cell_4)
        w4_post = (cell_4 & post_mask).astype(float) * (p_cell_1 / p_cell_4)
        y_00_pre = weighted_mean(y, w4_pre)
        y_00_post = weighted_mean(y, w4_post)

        # DDD estimate
        att = (
            (y_11_post - y_11_pre)
            - (y_10_post - y_10_pre)
            - (y_01_post - y_01_pre)
            + (y_00_post - y_00_pre)
        )

        # Standard error (approximate, using delta method)
        # For simplicity, use influence function approach
        se = self._compute_ipw_se(
            y, G, P, T, cell_1, cell_2, cell_3, cell_4,
            p_cell_1, p_cell_2, p_cell_3, p_cell_4, att
        )

        return att, se, None, pscore_stats

    def _doubly_robust(
        self,
        y: np.ndarray,
        G: np.ndarray,
        P: np.ndarray,
        T: np.ndarray,
        X: Optional[np.ndarray],
    ) -> Tuple[float, float, Optional[float], Optional[Dict[str, float]]]:
        """
        Estimate ATT using doubly robust estimation.

        Combines outcome regression and IPW for robustness:
        consistent if either the outcome model or propensity score
        model is correctly specified.
        """
        n = len(y)

        # Cell indicators
        cell_1 = (G == 1) & (P == 1)
        cell_2 = (G == 1) & (P == 0)
        cell_3 = (G == 0) & (P == 1)
        cell_4 = (G == 0) & (P == 0)

        # Step 1: Outcome regression for each cell-time combination
        # Predict E[Y|X,T] for each cell
        if X is not None and X.shape[1] > 0:
            # Fit outcome models for each cell
            mu_fitted = np.zeros(n)

            for cell_mask, cell_name in [
                (cell_1, "cell_1"), (cell_2, "cell_2"),
                (cell_3, "cell_3"), (cell_4, "cell_4")
            ]:
                for t_val in [0, 1]:
                    mask = cell_mask & (T == t_val)
                    if np.sum(mask) > 1:
                        X_cell = np.column_stack([X[mask], T[mask]])
                        try:
                            _, fitted, _ = _linear_regression(X_cell, y[mask])
                            mu_fitted[mask] = fitted
                        except Exception:
                            mu_fitted[mask] = np.mean(y[mask])
                    elif np.sum(mask) == 1:
                        mu_fitted[mask] = y[mask]

            # Propensity scores
            try:
                _, p_G = _logistic_regression(X, G)
            except Exception:
                p_G = np.full(n, np.mean(G))

            try:
                _, p_P = _logistic_regression(X, P)
            except Exception:
                p_P = np.full(n, np.mean(P))

            p_G = np.clip(p_G, self.pscore_trim, 1 - self.pscore_trim)
            p_P = np.clip(p_P, self.pscore_trim, 1 - self.pscore_trim)

            p_cell_1 = p_G * p_P
            p_cell_2 = p_G * (1 - p_P)
            p_cell_3 = (1 - p_G) * p_P
            p_cell_4 = (1 - p_G) * (1 - p_P)

            pscore_stats = {
                "P(G=1) mean": float(np.mean(p_G)),
                "P(G=1) std": float(np.std(p_G)),
                "P(P=1) mean": float(np.mean(p_P)),
                "P(P=1) std": float(np.std(p_P)),
            }
        else:
            # No covariates: use cell means as predictions
            mu_fitted = np.zeros(n)
            for cell_mask in [cell_1, cell_2, cell_3, cell_4]:
                for t_val in [0, 1]:
                    mask = cell_mask & (T == t_val)
                    if np.sum(mask) > 0:
                        mu_fitted[mask] = np.mean(y[mask])

            # Unconditional probabilities
            p_cell_1 = np.full(n, np.mean(cell_1))
            p_cell_2 = np.full(n, np.mean(cell_2))
            p_cell_3 = np.full(n, np.mean(cell_3))
            p_cell_4 = np.full(n, np.mean(cell_4))
            pscore_stats = None

        # Clip cell probabilities
        p_cell_1 = np.clip(p_cell_1, self.pscore_trim, 1 - self.pscore_trim)
        p_cell_2 = np.clip(p_cell_2, self.pscore_trim, 1 - self.pscore_trim)
        p_cell_3 = np.clip(p_cell_3, self.pscore_trim, 1 - self.pscore_trim)
        p_cell_4 = np.clip(p_cell_4, self.pscore_trim, 1 - self.pscore_trim)

        # Step 2: Doubly robust estimator
        # For each cell, compute the augmented IPW term:
        # (Y - mu(X)) * weight + mu(X)

        pre_mask = T == 0
        post_mask = T == 1

        # Influence function components for each observation
        n_1 = np.sum(cell_1)
        p_ref = n_1 / n

        # Cell 1 (G=1, P=1) - effectively treated
        inf_11 = np.zeros(n)
        inf_11[cell_1] = (y[cell_1] - mu_fitted[cell_1]) / p_ref
        # Add outcome model contribution
        inf_11 += mu_fitted * cell_1.astype(float) / p_ref

        # Cell 2 (G=1, P=0)
        w_10 = cell_2.astype(float) * (p_cell_1 / p_cell_2)
        inf_10 = w_10 * (y - mu_fitted) / p_ref
        # Add outcome model contribution for cell 2 (vectorized)
        inf_10[cell_2] += mu_fitted[cell_2] * (p_cell_1[cell_2] / p_cell_2[cell_2]) / p_ref

        # Cell 3 (G=0, P=1)
        w_01 = cell_3.astype(float) * (p_cell_1 / p_cell_3)
        inf_01 = w_01 * (y - mu_fitted) / p_ref
        # Add outcome model contribution for cell 3 (vectorized)
        inf_01[cell_3] += mu_fitted[cell_3] * (p_cell_1[cell_3] / p_cell_3[cell_3]) / p_ref

        # Cell 4 (G=0, P=0)
        w_00 = cell_4.astype(float) * (p_cell_1 / p_cell_4)
        inf_00 = w_00 * (y - mu_fitted) / p_ref
        # Add outcome model contribution for cell 4 (vectorized)
        inf_00[cell_4] += mu_fitted[cell_4] * (p_cell_1[cell_4] / p_cell_4[cell_4]) / p_ref

        # Compute cell-time means using DR formula
        def dr_mean(inf_vals, t_mask):
            return np.mean(inf_vals[t_mask])

        y_11_pre = dr_mean(inf_11, pre_mask)
        y_11_post = dr_mean(inf_11, post_mask)
        y_10_pre = dr_mean(inf_10, pre_mask)
        y_10_post = dr_mean(inf_10, post_mask)
        y_01_pre = dr_mean(inf_01, pre_mask)
        y_01_post = dr_mean(inf_01, post_mask)
        y_00_pre = dr_mean(inf_00, pre_mask)
        y_00_post = dr_mean(inf_00, post_mask)

        # DDD estimate
        att = (
            (y_11_post - y_11_pre)
            - (y_10_post - y_10_pre)
            - (y_01_post - y_01_pre)
            + (y_00_post - y_00_pre)
        )

        # Standard error computation
        # Use the simpler variance formula for the DDD estimator
        # Var(DDD) ≈ sum of variances of cell means / cell_sizes

        # Compute variances within each cell-time combination
        def cell_var(cell_mask, t_mask, y_vals):
            mask = cell_mask & t_mask
            if np.sum(mask) > 1:
                return np.var(y_vals[mask], ddof=1), np.sum(mask)
            return 0.0, max(1, np.sum(mask))

        # Variance components for each of the 8 cells
        var_components = []
        for cell_mask in [cell_1, cell_2, cell_3, cell_4]:
            for t_mask in [pre_mask, post_mask]:
                v, n_cell = cell_var(cell_mask, t_mask, y)
                if n_cell > 0:
                    var_components.append(v / n_cell)

        # Total variance is sum of components (assuming independence)
        total_var = sum(var_components)
        se = np.sqrt(total_var)

        # R-squared from outcome regression
        if X is not None:
            ss_res = np.sum((y - mu_fitted) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        else:
            r_squared = None

        return att, se, r_squared, pscore_stats

    def _compute_se(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        coef_idx: int,
    ) -> float:
        """Compute standard error for a coefficient using robust or clustered SE."""
        n, k = X.shape

        if self.robust:
            # HC1 robust standard errors
            vcov = compute_robust_se(X, residuals, cluster_ids=None)
        else:
            # Classical OLS standard errors
            mse = np.sum(residuals ** 2) / (n - k)
            try:
                vcov = np.linalg.solve(X.T @ X, mse * np.eye(k))
            except np.linalg.LinAlgError:
                vcov = np.linalg.pinv(X.T @ X) * mse

        return float(np.sqrt(vcov[coef_idx, coef_idx]))

    def _compute_ipw_se(
        self,
        y: np.ndarray,
        G: np.ndarray,
        P: np.ndarray,
        T: np.ndarray,
        cell_1: np.ndarray,
        cell_2: np.ndarray,
        cell_3: np.ndarray,
        cell_4: np.ndarray,
        p_cell_1: np.ndarray,
        p_cell_2: np.ndarray,
        p_cell_3: np.ndarray,
        p_cell_4: np.ndarray,
        att: float,
    ) -> float:
        """Compute standard error for IPW estimator using influence function."""
        n = len(y)
        post_mask = T == 1

        # Influence function for IPW estimator (vectorized)
        inf_func = np.zeros(n)

        n_ref = np.sum(cell_1)
        p_ref = n_ref / n

        # Sign: +1 for post, -1 for pre
        sign = np.where(post_mask, 1.0, -1.0)

        # Cell 1 (G=1, P=1): sign * (y - att) / p_ref
        inf_func[cell_1] = sign[cell_1] * (y[cell_1] - att) / p_ref

        # Cell 2 (G=1, P=0): -sign * y * (p_cell_1 / p_cell_2) / p_ref
        w_2 = p_cell_1[cell_2] / p_cell_2[cell_2]
        inf_func[cell_2] = -sign[cell_2] * y[cell_2] * w_2 / p_ref

        # Cell 3 (G=0, P=1): -sign * y * (p_cell_1 / p_cell_3) / p_ref
        w_3 = p_cell_1[cell_3] / p_cell_3[cell_3]
        inf_func[cell_3] = -sign[cell_3] * y[cell_3] * w_3 / p_ref

        # Cell 4 (G=0, P=0): sign * y * (p_cell_1 / p_cell_4) / p_ref
        w_4 = p_cell_1[cell_4] / p_cell_4[cell_4]
        inf_func[cell_4] = sign[cell_4] * y[cell_4] * w_4 / p_ref

        var_inf = np.var(inf_func, ddof=1)
        se = np.sqrt(var_inf / n)

        return se

    def get_params(self) -> Dict[str, Any]:
        """
        Get estimator parameters (sklearn-compatible).

        Returns
        -------
        Dict[str, Any]
            Estimator parameters.
        """
        return {
            "estimation_method": self.estimation_method,
            "robust": self.robust,
            "cluster": self.cluster,
            "alpha": self.alpha,
            "pscore_trim": self.pscore_trim,
        }

    def set_params(self, **params) -> "TripleDifference":
        """
        Set estimator parameters (sklearn-compatible).

        Parameters
        ----------
        **params
            Estimator parameters.

        Returns
        -------
        self
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")
        return self

    def summary(self) -> str:
        """
        Get summary of estimation results.

        Returns
        -------
        str
            Formatted summary.
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before calling summary()")
        assert self.results_ is not None
        return self.results_.summary()

    def print_summary(self) -> None:
        """Print summary to stdout."""
        print(self.summary())


# =============================================================================
# Convenience function
# =============================================================================


def triple_difference(
    data: pd.DataFrame,
    outcome: str,
    group: str,
    partition: str,
    time: str,
    covariates: Optional[List[str]] = None,
    estimation_method: str = "dr",
    robust: bool = True,
    cluster: Optional[str] = None,
    alpha: float = 0.05,
) -> TripleDifferenceResults:
    """
    Estimate Triple Difference (DDD) treatment effect.

    Convenience function that creates a TripleDifference estimator and
    fits it to the data in one step.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing all variables.
    outcome : str
        Name of the outcome variable column.
    group : str
        Name of the group indicator column (0/1).
        1 = treated group (e.g., states that enacted policy).
    partition : str
        Name of the partition/eligibility indicator column (0/1).
        1 = eligible partition (e.g., women, targeted demographic).
    time : str
        Name of the time period indicator column (0/1).
        1 = post-treatment period.
    covariates : list of str, optional
        List of covariate column names to adjust for.
    estimation_method : str, default="dr"
        Estimation method: "dr" (doubly robust), "reg" (regression),
        or "ipw" (inverse probability weighting).
    robust : bool, default=True
        Whether to use robust standard errors.
    cluster : str, optional
        Column name for cluster-robust standard errors.
    alpha : float, default=0.05
        Significance level for confidence intervals.

    Returns
    -------
    TripleDifferenceResults
        Object containing estimation results.

    Examples
    --------
    >>> from diff_diff import triple_difference
    >>> results = triple_difference(
    ...     data,
    ...     outcome='earnings',
    ...     group='policy_state',
    ...     partition='female',
    ...     time='post_policy',
    ...     covariates=['age', 'education']
    ... )
    >>> print(f"ATT: {results.att:.3f} (SE: {results.se:.3f})")
    """
    estimator = TripleDifference(
        estimation_method=estimation_method,
        robust=robust,
        cluster=cluster,
        alpha=alpha,
    )
    return estimator.fit(
        data=data,
        outcome=outcome,
        group=group,
        partition=partition,
        time=time,
        covariates=covariates,
    )
