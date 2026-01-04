"""
Staggered Difference-in-Differences estimators.

Implements modern methods for DiD with variation in treatment timing,
including the Callaway-Sant'Anna (2021) estimator.
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
)

# =============================================================================
# Bootstrap Weight Generators
# =============================================================================


def _generate_bootstrap_weights(
    n_units: int,
    weight_type: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate bootstrap weights for multiplier bootstrap.

    Parameters
    ----------
    n_units : int
        Number of units (clusters) to generate weights for.
    weight_type : str
        Type of weights: "rademacher", "mammen", or "webb".
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Array of bootstrap weights with shape (n_units,).
    """
    if weight_type == "rademacher":
        # Rademacher: +1 or -1 with equal probability
        return rng.choice([-1.0, 1.0], size=n_units)

    elif weight_type == "mammen":
        # Mammen's two-point distribution
        # E[v] = 0, E[v^2] = 1, E[v^3] = 1
        sqrt5 = np.sqrt(5)
        val1 = -(sqrt5 - 1) / 2  # ≈ -0.618
        val2 = (sqrt5 + 1) / 2   # ≈ 1.618 (golden ratio)
        p1 = (sqrt5 + 1) / (2 * sqrt5)  # ≈ 0.724
        return rng.choice([val1, val2], size=n_units, p=[p1, 1 - p1])

    elif weight_type == "webb":
        # Webb's 6-point distribution (recommended for few clusters)
        values = np.array([
            -np.sqrt(3 / 2), -np.sqrt(2 / 2), -np.sqrt(1 / 2),
            np.sqrt(1 / 2), np.sqrt(2 / 2), np.sqrt(3 / 2)
        ])
        probs = np.array([1, 2, 3, 3, 2, 1]) / 12
        return rng.choice(values, size=n_units, p=probs)

    else:
        raise ValueError(
            f"weight_type must be 'rademacher', 'mammen', or 'webb', "
            f"got '{weight_type}'"
        )


# =============================================================================
# Bootstrap Results Container
# =============================================================================


@dataclass
class CSBootstrapResults:
    """
    Results from Callaway-Sant'Anna multiplier bootstrap inference.

    Attributes
    ----------
    n_bootstrap : int
        Number of bootstrap iterations.
    weight_type : str
        Type of bootstrap weights used.
    alpha : float
        Significance level used for confidence intervals.
    overall_att_se : float
        Bootstrap standard error for overall ATT.
    overall_att_ci : Tuple[float, float]
        Bootstrap confidence interval for overall ATT.
    overall_att_p_value : float
        Bootstrap p-value for overall ATT.
    group_time_ses : Dict[Tuple[Any, Any], float]
        Bootstrap SEs for each ATT(g,t).
    group_time_cis : Dict[Tuple[Any, Any], Tuple[float, float]]
        Bootstrap CIs for each ATT(g,t).
    group_time_p_values : Dict[Tuple[Any, Any], float]
        Bootstrap p-values for each ATT(g,t).
    event_study_ses : Optional[Dict[int, float]]
        Bootstrap SEs for event study effects.
    event_study_cis : Optional[Dict[int, Tuple[float, float]]]
        Bootstrap CIs for event study effects.
    event_study_p_values : Optional[Dict[int, float]]
        Bootstrap p-values for event study effects.
    group_effect_ses : Optional[Dict[Any, float]]
        Bootstrap SEs for group effects.
    group_effect_cis : Optional[Dict[Any, Tuple[float, float]]]
        Bootstrap CIs for group effects.
    group_effect_p_values : Optional[Dict[Any, float]]
        Bootstrap p-values for group effects.
    bootstrap_distribution : Optional[np.ndarray]
        Full bootstrap distribution of overall ATT (if requested).
    """
    n_bootstrap: int
    weight_type: str
    alpha: float
    overall_att_se: float
    overall_att_ci: Tuple[float, float]
    overall_att_p_value: float
    group_time_ses: Dict[Tuple[Any, Any], float]
    group_time_cis: Dict[Tuple[Any, Any], Tuple[float, float]]
    group_time_p_values: Dict[Tuple[Any, Any], float]
    event_study_ses: Optional[Dict[int, float]] = None
    event_study_cis: Optional[Dict[int, Tuple[float, float]]] = None
    event_study_p_values: Optional[Dict[int, float]] = None
    group_effect_ses: Optional[Dict[Any, float]] = None
    group_effect_cis: Optional[Dict[Any, Tuple[float, float]]] = None
    group_effect_p_values: Optional[Dict[Any, float]] = None
    bootstrap_distribution: Optional[np.ndarray] = field(default=None, repr=False)


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
    # Add intercept
    X_with_intercept = np.column_stack([np.ones(n), X])

    def neg_log_likelihood(beta):
        z = X_with_intercept @ beta
        # Clip to prevent overflow
        z = np.clip(z, -500, 500)
        log_lik = np.sum(y * z - np.log(1 + np.exp(z)))
        return -log_lik

    def gradient(beta):
        z = X_with_intercept @ beta
        z = np.clip(z, -500, 500)
        probs = 1 / (1 + np.exp(-z))
        return -X_with_intercept.T @ (y - probs)

    # Initialize with zeros
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
) -> Tuple[np.ndarray, np.ndarray]:
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
    residuals : np.ndarray
        Residuals from the fit.
    """
    n = X.shape[0]
    # Add intercept
    X_with_intercept = np.column_stack([np.ones(n), X])

    # OLS: beta = (X'X)^{-1} X'y
    try:
        beta = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        # Fallback: use pseudo-inverse
        beta = np.linalg.pinv(X_with_intercept) @ y

    fitted = X_with_intercept @ beta
    residuals = y - fitted

    return beta, residuals


@dataclass
class GroupTimeEffect:
    """
    Treatment effect for a specific group-time combination.

    Attributes
    ----------
    group : any
        The treatment cohort (first treatment period).
    time : any
        The time period.
    effect : float
        The ATT(g,t) estimate.
    se : float
        Standard error.
    n_treated : int
        Number of treated observations.
    n_control : int
        Number of control observations.
    """
    group: Any
    time: Any
    effect: float
    se: float
    t_stat: float
    p_value: float
    conf_int: Tuple[float, float]
    n_treated: int
    n_control: int

    @property
    def is_significant(self) -> bool:
        """Check if effect is significant at 0.05 level."""
        return bool(self.p_value < 0.05)

    @property
    def significance_stars(self) -> str:
        """Return significance stars based on p-value."""
        return _get_significance_stars(self.p_value)


@dataclass
class CallawaySantAnnaResults:
    """
    Results from Callaway-Sant'Anna (2021) staggered DiD estimation.

    This class stores group-time average treatment effects ATT(g,t) and
    provides methods for aggregation into summary measures.

    Attributes
    ----------
    group_time_effects : dict
        Dictionary mapping (group, time) tuples to effect dictionaries.
    overall_att : float
        Overall average treatment effect (weighted average of ATT(g,t)).
    overall_se : float
        Standard error of overall ATT.
    overall_p_value : float
        P-value for overall ATT.
    overall_conf_int : tuple
        Confidence interval for overall ATT.
    groups : list
        List of treatment cohorts (first treatment periods).
    time_periods : list
        List of all time periods.
    n_obs : int
        Total number of observations.
    n_treated_units : int
        Number of ever-treated units.
    n_control_units : int
        Number of never-treated units.
    event_study_effects : dict, optional
        Effects aggregated by relative time (event study).
    group_effects : dict, optional
        Effects aggregated by treatment cohort.
    """
    group_time_effects: Dict[Tuple[Any, Any], Dict[str, Any]]
    overall_att: float
    overall_se: float
    overall_t_stat: float
    overall_p_value: float
    overall_conf_int: Tuple[float, float]
    groups: List[Any]
    time_periods: List[Any]
    n_obs: int
    n_treated_units: int
    n_control_units: int
    alpha: float = 0.05
    control_group: str = "never_treated"
    event_study_effects: Optional[Dict[int, Dict[str, Any]]] = field(default=None)
    group_effects: Optional[Dict[Any, Dict[str, Any]]] = field(default=None)
    influence_functions: Optional[np.ndarray] = field(default=None, repr=False)
    bootstrap_results: Optional[CSBootstrapResults] = field(default=None, repr=False)

    def __repr__(self) -> str:
        """Concise string representation."""
        sig = _get_significance_stars(self.overall_p_value)
        return (
            f"CallawaySantAnnaResults(ATT={self.overall_att:.4f}{sig}, "
            f"SE={self.overall_se:.4f}, "
            f"n_groups={len(self.groups)}, "
            f"n_periods={len(self.time_periods)})"
        )

    def summary(self, alpha: Optional[float] = None) -> str:
        """
        Generate formatted summary of estimation results.

        Parameters
        ----------
        alpha : float, optional
            Significance level. Defaults to alpha used in estimation.

        Returns
        -------
        str
            Formatted summary.
        """
        alpha = alpha or self.alpha
        conf_level = int((1 - alpha) * 100)

        lines = [
            "=" * 85,
            "Callaway-Sant'Anna Staggered Difference-in-Differences Results".center(85),
            "=" * 85,
            "",
            f"{'Total observations:':<30} {self.n_obs:>10}",
            f"{'Treated units:':<30} {self.n_treated_units:>10}",
            f"{'Control units:':<30} {self.n_control_units:>10}",
            f"{'Treatment cohorts:':<30} {len(self.groups):>10}",
            f"{'Time periods:':<30} {len(self.time_periods):>10}",
            f"{'Control group:':<30} {self.control_group:>10}",
            "",
        ]

        # Overall ATT
        lines.extend([
            "-" * 85,
            "Overall Average Treatment Effect on the Treated".center(85),
            "-" * 85,
            f"{'Parameter':<15} {'Estimate':>12} {'Std. Err.':>12} {'t-stat':>10} {'P>|t|':>10} {'Sig.':>6}",
            "-" * 85,
            f"{'ATT':<15} {self.overall_att:>12.4f} {self.overall_se:>12.4f} "
            f"{self.overall_t_stat:>10.3f} {self.overall_p_value:>10.4f} "
            f"{_get_significance_stars(self.overall_p_value):>6}",
            "-" * 85,
            "",
            f"{conf_level}% Confidence Interval: [{self.overall_conf_int[0]:.4f}, {self.overall_conf_int[1]:.4f}]",
            "",
        ])

        # Event study effects if available
        if self.event_study_effects:
            lines.extend([
                "-" * 85,
                "Event Study (Dynamic) Effects".center(85),
                "-" * 85,
                f"{'Rel. Period':<15} {'Estimate':>12} {'Std. Err.':>12} {'t-stat':>10} {'P>|t|':>10} {'Sig.':>6}",
                "-" * 85,
            ])

            for rel_t in sorted(self.event_study_effects.keys()):
                eff = self.event_study_effects[rel_t]
                sig = _get_significance_stars(eff['p_value'])
                lines.append(
                    f"{rel_t:<15} {eff['effect']:>12.4f} {eff['se']:>12.4f} "
                    f"{eff['t_stat']:>10.3f} {eff['p_value']:>10.4f} {sig:>6}"
                )

            lines.extend(["-" * 85, ""])

        # Group effects if available
        if self.group_effects:
            lines.extend([
                "-" * 85,
                "Effects by Treatment Cohort".center(85),
                "-" * 85,
                f"{'Cohort':<15} {'Estimate':>12} {'Std. Err.':>12} {'t-stat':>10} {'P>|t|':>10} {'Sig.':>6}",
                "-" * 85,
            ])

            for group in sorted(self.group_effects.keys()):
                eff = self.group_effects[group]
                sig = _get_significance_stars(eff['p_value'])
                lines.append(
                    f"{group:<15} {eff['effect']:>12.4f} {eff['se']:>12.4f} "
                    f"{eff['t_stat']:>10.3f} {eff['p_value']:>10.4f} {sig:>6}"
                )

            lines.extend(["-" * 85, ""])

        lines.extend([
            "Signif. codes: '***' 0.001, '**' 0.01, '*' 0.05, '.' 0.1",
            "=" * 85,
        ])

        return "\n".join(lines)

    def print_summary(self, alpha: Optional[float] = None) -> None:
        """Print summary to stdout."""
        print(self.summary(alpha))

    def to_dataframe(self, level: str = "group_time") -> pd.DataFrame:
        """
        Convert results to DataFrame.

        Parameters
        ----------
        level : str, default="group_time"
            Level of aggregation: "group_time", "event_study", or "group".

        Returns
        -------
        pd.DataFrame
            Results as DataFrame.
        """
        if level == "group_time":
            rows = []
            for (g, t), data in self.group_time_effects.items():
                rows.append({
                    'group': g,
                    'time': t,
                    'effect': data['effect'],
                    'se': data['se'],
                    't_stat': data['t_stat'],
                    'p_value': data['p_value'],
                    'conf_int_lower': data['conf_int'][0],
                    'conf_int_upper': data['conf_int'][1],
                })
            return pd.DataFrame(rows)

        elif level == "event_study":
            if self.event_study_effects is None:
                raise ValueError("Event study effects not computed. Use aggregate='event_study'.")
            rows = []
            for rel_t, data in sorted(self.event_study_effects.items()):
                rows.append({
                    'relative_period': rel_t,
                    'effect': data['effect'],
                    'se': data['se'],
                    't_stat': data['t_stat'],
                    'p_value': data['p_value'],
                    'conf_int_lower': data['conf_int'][0],
                    'conf_int_upper': data['conf_int'][1],
                })
            return pd.DataFrame(rows)

        elif level == "group":
            if self.group_effects is None:
                raise ValueError("Group effects not computed. Use aggregate='group'.")
            rows = []
            for group, data in sorted(self.group_effects.items()):
                rows.append({
                    'group': group,
                    'effect': data['effect'],
                    'se': data['se'],
                    't_stat': data['t_stat'],
                    'p_value': data['p_value'],
                    'conf_int_lower': data['conf_int'][0],
                    'conf_int_upper': data['conf_int'][1],
                })
            return pd.DataFrame(rows)

        else:
            raise ValueError(f"Unknown level: {level}. Use 'group_time', 'event_study', or 'group'.")

    @property
    def is_significant(self) -> bool:
        """Check if overall ATT is significant."""
        return bool(self.overall_p_value < self.alpha)

    @property
    def significance_stars(self) -> str:
        """Significance stars for overall ATT."""
        return _get_significance_stars(self.overall_p_value)


class CallawaySantAnna:
    """
    Callaway-Sant'Anna (2021) estimator for staggered Difference-in-Differences.

    This estimator handles DiD designs with variation in treatment timing
    (staggered adoption) and heterogeneous treatment effects. It avoids the
    bias of traditional two-way fixed effects (TWFE) estimators by:

    1. Computing group-time average treatment effects ATT(g,t) for each
       cohort g (units first treated in period g) and time t.
    2. Aggregating these to summary measures (overall ATT, event study, etc.)
       using appropriate weights.

    Parameters
    ----------
    control_group : str, default="never_treated"
        Which units to use as controls:
        - "never_treated": Use only never-treated units (recommended)
        - "not_yet_treated": Use never-treated and not-yet-treated units
    anticipation : int, default=0
        Number of periods before treatment where effects may occur.
        Set to > 0 if treatment effects can begin before the official
        treatment date.
    estimation_method : str, default="dr"
        Estimation method:
        - "dr": Doubly robust (recommended)
        - "ipw": Inverse probability weighting
        - "reg": Outcome regression
    alpha : float, default=0.05
        Significance level for confidence intervals.
    cluster : str, optional
        Column name for cluster-robust standard errors.
        Defaults to unit-level clustering.
    n_bootstrap : int, default=0
        Number of bootstrap iterations for inference.
        If 0, uses analytical standard errors.
        Recommended: 999 or more for reliable inference.
    bootstrap_weights : str, default="rademacher"
        Type of weights for multiplier bootstrap:
        - "rademacher": +1/-1 with equal probability (standard choice)
        - "mammen": Two-point distribution (asymptotically valid, matches skewness)
        - "webb": Six-point distribution (recommended when n_clusters < 20)
    bootstrap_weight_type : str, optional
        .. deprecated:: 1.0.1
            Use ``bootstrap_weights`` instead. Will be removed in v2.0.
    seed : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    results_ : CallawaySantAnnaResults
        Estimation results after calling fit().
    is_fitted_ : bool
        Whether the model has been fitted.

    Examples
    --------
    Basic usage:

    >>> import pandas as pd
    >>> from diff_diff import CallawaySantAnna
    >>>
    >>> # Panel data with staggered treatment
    >>> # 'first_treat' = period when unit was first treated (0 if never treated)
    >>> data = pd.DataFrame({
    ...     'unit': [...],
    ...     'time': [...],
    ...     'outcome': [...],
    ...     'first_treat': [...]  # 0 for never-treated, else first treatment period
    ... })
    >>>
    >>> cs = CallawaySantAnna()
    >>> results = cs.fit(data, outcome='outcome', unit='unit',
    ...                  time='time', first_treat='first_treat')
    >>>
    >>> results.print_summary()

    With event study aggregation:

    >>> cs = CallawaySantAnna()
    >>> results = cs.fit(data, outcome='outcome', unit='unit',
    ...                  time='time', first_treat='first_treat',
    ...                  aggregate='event_study')
    >>>
    >>> # Plot event study
    >>> from diff_diff import plot_event_study
    >>> plot_event_study(results)

    With covariate adjustment (conditional parallel trends):

    >>> # When parallel trends only holds conditional on covariates
    >>> cs = CallawaySantAnna(estimation_method='dr')  # doubly robust
    >>> results = cs.fit(data, outcome='outcome', unit='unit',
    ...                  time='time', first_treat='first_treat',
    ...                  covariates=['age', 'income'])
    >>>
    >>> # DR is recommended: consistent if either outcome model
    >>> # or propensity model is correctly specified

    Notes
    -----
    The key innovation of Callaway & Sant'Anna (2021) is the disaggregated
    approach: instead of estimating a single treatment effect, they estimate
    ATT(g,t) for each cohort-time pair. This avoids the "forbidden comparison"
    problem where already-treated units act as controls.

    The ATT(g,t) is identified under parallel trends conditional on covariates:

        E[Y(0)_t - Y(0)_g-1 | G=g] = E[Y(0)_t - Y(0)_g-1 | C=1]

    where G=g indicates treatment cohort g and C=1 indicates control units.

    References
    ----------
    Callaway, B., & Sant'Anna, P. H. (2021). Difference-in-Differences with
    multiple time periods. Journal of Econometrics, 225(2), 200-230.
    """

    def __init__(
        self,
        control_group: str = "never_treated",
        anticipation: int = 0,
        estimation_method: str = "dr",
        alpha: float = 0.05,
        cluster: Optional[str] = None,
        n_bootstrap: int = 0,
        bootstrap_weights: Optional[str] = None,
        bootstrap_weight_type: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        import warnings

        if control_group not in ["never_treated", "not_yet_treated"]:
            raise ValueError(
                f"control_group must be 'never_treated' or 'not_yet_treated', "
                f"got '{control_group}'"
            )
        if estimation_method not in ["dr", "ipw", "reg"]:
            raise ValueError(
                f"estimation_method must be 'dr', 'ipw', or 'reg', "
                f"got '{estimation_method}'"
            )

        # Handle bootstrap_weight_type deprecation
        if bootstrap_weight_type is not None:
            warnings.warn(
                "bootstrap_weight_type is deprecated and will be removed in v2.0. "
                "Use bootstrap_weights instead.",
                DeprecationWarning,
                stacklevel=2
            )
            if bootstrap_weights is None:
                bootstrap_weights = bootstrap_weight_type

        # Default to rademacher if neither specified
        if bootstrap_weights is None:
            bootstrap_weights = "rademacher"

        if bootstrap_weights not in ["rademacher", "mammen", "webb"]:
            raise ValueError(
                f"bootstrap_weights must be 'rademacher', 'mammen', or 'webb', "
                f"got '{bootstrap_weights}'"
            )

        self.control_group = control_group
        self.anticipation = anticipation
        self.estimation_method = estimation_method
        self.alpha = alpha
        self.cluster = cluster
        self.n_bootstrap = n_bootstrap
        self.bootstrap_weights = bootstrap_weights
        # Keep bootstrap_weight_type for backward compatibility
        self.bootstrap_weight_type = bootstrap_weights
        self.seed = seed

        self.is_fitted_ = False
        self.results_ = None

    def fit(
        self,
        data: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        first_treat: str,
        covariates: Optional[List[str]] = None,
        aggregate: Optional[str] = None,
        balance_e: Optional[int] = None,
    ) -> CallawaySantAnnaResults:
        """
        Fit the Callaway-Sant'Anna estimator.

        Parameters
        ----------
        data : pd.DataFrame
            Panel data with unit and time identifiers.
        outcome : str
            Name of outcome variable column.
        unit : str
            Name of unit identifier column.
        time : str
            Name of time period column.
        first_treat : str
            Name of column indicating when unit was first treated.
            Use 0 (or np.inf) for never-treated units.
        covariates : list, optional
            List of covariate column names for conditional parallel trends.
        aggregate : str, optional
            How to aggregate group-time effects:
            - None: Only compute ATT(g,t) (default)
            - "simple": Simple weighted average (overall ATT)
            - "event_study": Aggregate by relative time (event study)
            - "group": Aggregate by treatment cohort
            - "all": Compute all aggregations
        balance_e : int, optional
            For event study, balance the panel at relative time e.
            Ensures all groups contribute to each relative period.

        Returns
        -------
        CallawaySantAnnaResults
            Object containing all estimation results.

        Raises
        ------
        ValueError
            If required columns are missing or data validation fails.
        """
        # Validate inputs
        required_cols = [outcome, unit, time, first_treat]
        if covariates:
            required_cols.extend(covariates)

        missing = [c for c in required_cols if c not in data.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # Create working copy
        df = data.copy()

        # Ensure numeric types
        df[time] = pd.to_numeric(df[time])
        df[first_treat] = pd.to_numeric(df[first_treat])

        # Identify groups and time periods
        time_periods = sorted(df[time].unique())
        treatment_groups = sorted([g for g in df[first_treat].unique() if g > 0])

        # Never-treated indicator (first_treat = 0 or inf)
        df['_never_treated'] = (df[first_treat] == 0) | (df[first_treat] == np.inf)

        # Get unique units
        unit_info = df.groupby(unit).agg({
            first_treat: 'first',
            '_never_treated': 'first'
        }).reset_index()

        n_treated_units = (unit_info[first_treat] > 0).sum()
        n_control_units = (unit_info['_never_treated']).sum()

        if n_control_units == 0:
            raise ValueError("No never-treated units found. Check 'first_treat' column.")

        # Compute ATT(g,t) for each group-time combination
        group_time_effects = {}
        influence_func_info = {}  # Store influence functions for bootstrap

        for g in treatment_groups:
            # Periods for which we compute effects (t >= g - anticipation)
            valid_periods = [t for t in time_periods if t >= g - self.anticipation]

            for t in valid_periods:
                att_gt, se_gt, n_treat, n_ctrl, inf_info = self._compute_att_gt(
                    df, outcome, unit, time, first_treat, g, t,
                    covariates, time_periods
                )

                if att_gt is not None:
                    t_stat = att_gt / se_gt if se_gt > 0 else 0.0
                    p_val = compute_p_value(t_stat)
                    ci = compute_confidence_interval(att_gt, se_gt, self.alpha)

                    group_time_effects[(g, t)] = {
                        'effect': att_gt,
                        'se': se_gt,
                        't_stat': t_stat,
                        'p_value': p_val,
                        'conf_int': ci,
                        'n_treated': n_treat,
                        'n_control': n_ctrl,
                    }

                    if inf_info is not None:
                        influence_func_info[(g, t)] = inf_info

        if not group_time_effects:
            raise ValueError(
                "Could not estimate any group-time effects. "
                "Check that data has sufficient observations."
            )

        # Compute overall ATT (simple aggregation)
        overall_att, overall_se = self._aggregate_simple(group_time_effects, df, unit)
        overall_t = overall_att / overall_se if overall_se > 0 else 0.0
        overall_p = compute_p_value(overall_t)
        overall_ci = compute_confidence_interval(overall_att, overall_se, self.alpha)

        # Compute additional aggregations if requested
        event_study_effects = None
        group_effects = None

        if aggregate in ["event_study", "all"]:
            event_study_effects = self._aggregate_event_study(
                group_time_effects, treatment_groups, time_periods, balance_e
            )

        if aggregate in ["group", "all"]:
            group_effects = self._aggregate_by_group(
                group_time_effects, treatment_groups
            )

        # Run bootstrap inference if requested
        bootstrap_results = None
        if self.n_bootstrap > 0 and influence_func_info:
            bootstrap_results = self._run_multiplier_bootstrap(
                group_time_effects=group_time_effects,
                influence_func_info=influence_func_info,
                aggregate=aggregate,
                balance_e=balance_e,
                treatment_groups=treatment_groups,
                time_periods=time_periods,
            )

            # Update estimates with bootstrap inference
            overall_se = bootstrap_results.overall_att_se
            overall_t = overall_att / overall_se if overall_se > 0 else 0.0
            overall_p = bootstrap_results.overall_att_p_value
            overall_ci = bootstrap_results.overall_att_ci

            # Update group-time effects with bootstrap SEs
            for gt in group_time_effects:
                if gt in bootstrap_results.group_time_ses:
                    group_time_effects[gt]['se'] = bootstrap_results.group_time_ses[gt]
                    group_time_effects[gt]['conf_int'] = bootstrap_results.group_time_cis[gt]
                    group_time_effects[gt]['p_value'] = bootstrap_results.group_time_p_values[gt]
                    effect = float(group_time_effects[gt]['effect'])
                    se = float(group_time_effects[gt]['se'])
                    group_time_effects[gt]['t_stat'] = effect / se if se > 0 else 0.0

            # Update event study effects with bootstrap SEs
            if (event_study_effects is not None
                and bootstrap_results.event_study_ses is not None
                and bootstrap_results.event_study_cis is not None
                and bootstrap_results.event_study_p_values is not None):
                for e in event_study_effects:
                    if e in bootstrap_results.event_study_ses:
                        event_study_effects[e]['se'] = bootstrap_results.event_study_ses[e]
                        event_study_effects[e]['conf_int'] = bootstrap_results.event_study_cis[e]
                        p_val = bootstrap_results.event_study_p_values[e]
                        event_study_effects[e]['p_value'] = p_val
                        effect = float(event_study_effects[e]['effect'])
                        se = float(event_study_effects[e]['se'])
                        event_study_effects[e]['t_stat'] = effect / se if se > 0 else 0.0

            # Update group effects with bootstrap SEs
            if (group_effects is not None
                and bootstrap_results.group_effect_ses is not None
                and bootstrap_results.group_effect_cis is not None
                and bootstrap_results.group_effect_p_values is not None):
                for g in group_effects:
                    if g in bootstrap_results.group_effect_ses:
                        group_effects[g]['se'] = bootstrap_results.group_effect_ses[g]
                        group_effects[g]['conf_int'] = bootstrap_results.group_effect_cis[g]
                        group_effects[g]['p_value'] = bootstrap_results.group_effect_p_values[g]
                        effect = float(group_effects[g]['effect'])
                        se = float(group_effects[g]['se'])
                        group_effects[g]['t_stat'] = effect / se if se > 0 else 0.0

        # Store results
        self.results_ = CallawaySantAnnaResults(
            group_time_effects=group_time_effects,
            overall_att=overall_att,
            overall_se=overall_se,
            overall_t_stat=overall_t,
            overall_p_value=overall_p,
            overall_conf_int=overall_ci,
            groups=treatment_groups,
            time_periods=time_periods,
            n_obs=len(df),
            n_treated_units=n_treated_units,
            n_control_units=n_control_units,
            alpha=self.alpha,
            control_group=self.control_group,
            event_study_effects=event_study_effects,
            group_effects=group_effects,
            bootstrap_results=bootstrap_results,
        )

        self.is_fitted_ = True
        return self.results_

    def _compute_att_gt(
        self,
        df: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        first_treat: str,
        g: Any,
        t: Any,
        covariates: Optional[List[str]],
        all_periods: List[Any],
    ) -> Tuple[Optional[float], float, int, int, Optional[Dict[str, Any]]]:
        """
        Compute ATT(g,t) for a specific group-time combination.

        Uses 2x2 DiD comparing:
        - Treated: Units in cohort g
        - Control: Never-treated units (or not-yet-treated if specified)
        - Pre-period: g - 1 (or earlier if anticipation > 0)
        - Post-period: t
        """
        # Base period for comparison
        base_period = g - 1 - self.anticipation
        if base_period not in all_periods:
            # Find closest earlier period
            earlier = [p for p in all_periods if p < g - self.anticipation]
            if not earlier:
                return None, 0.0, 0, 0, None
            base_period = max(earlier)

        # Treated group: units first treated in period g
        treated_units = df[df[first_treat] == g][unit].unique()

        # Control group
        if self.control_group == "never_treated":
            control_mask = df['_never_treated']
        else:  # not_yet_treated
            # Not yet treated at time t
            control_mask = (df['_never_treated']) | (df[first_treat] > t)

        control_units = df[control_mask][unit].unique()

        if len(treated_units) == 0 or len(control_units) == 0:
            return None, 0.0, 0, 0, None

        # Get data for the two periods
        df_base = df[df[time] == base_period].set_index(unit)
        df_post = df[df[time] == t].set_index(unit)

        # Compute outcome changes for treated
        treated_base = df_base.loc[df_base.index.isin(treated_units), outcome]
        treated_post = df_post.loc[df_post.index.isin(treated_units), outcome]
        treated_common = treated_base.index.intersection(treated_post.index)

        if len(treated_common) == 0:
            return None, 0.0, 0, 0, None

        treated_change = (
            treated_post.loc[treated_common].values -
            treated_base.loc[treated_common].values
        )

        # Compute outcome changes for control
        control_base = df_base.loc[df_base.index.isin(control_units), outcome]
        control_post = df_post.loc[df_post.index.isin(control_units), outcome]
        control_common = control_base.index.intersection(control_post.index)

        if len(control_common) == 0:
            return None, 0.0, 0, 0, None

        control_change = (
            control_post.loc[control_common].values -
            control_base.loc[control_common].values
        )

        # Get covariates if specified (use base period values for conditioning)
        X_treated = None
        X_control = None
        if covariates:
            try:
                X_treated = df_base.loc[treated_common, covariates].values
                X_control = df_base.loc[control_common, covariates].values
                # Check for missing values and handle them
                if np.any(np.isnan(X_treated)) or np.any(np.isnan(X_control)):
                    warnings.warn(
                        f"Missing values in covariates for group {g}, time {t}. "
                        "Falling back to unconditional estimation.",
                        UserWarning,
                        stacklevel=3,
                    )
                    X_treated = None
                    X_control = None
            except KeyError:
                warnings.warn(
                    f"Could not extract covariates for group {g}, time {t}. "
                    "Falling back to unconditional estimation.",
                    UserWarning,
                    stacklevel=3,
                )
                X_treated = None
                X_control = None

        # Estimation method
        if self.estimation_method == "reg":
            att_gt, se_gt, inf_func = self._outcome_regression(
                treated_change, control_change, X_treated, X_control
            )
        elif self.estimation_method == "ipw":
            att_gt, se_gt, inf_func = self._ipw_estimation(
                treated_change, control_change,
                len(treated_common), len(control_common),
                X_treated, X_control
            )
        else:  # doubly robust
            att_gt, se_gt, inf_func = self._doubly_robust(
                treated_change, control_change, X_treated, X_control
            )

        # Package influence function info with unit IDs for bootstrap
        n_t = len(treated_common)
        inf_func_info = {
            'treated_units': list(treated_common),
            'control_units': list(control_common),
            'treated_inf': inf_func[:n_t],
            'control_inf': inf_func[n_t:],
        }

        return att_gt, se_gt, len(treated_common), len(control_common), inf_func_info

    def _outcome_regression(
        self,
        treated_change: np.ndarray,
        control_change: np.ndarray,
        X_treated: Optional[np.ndarray] = None,
        X_control: Optional[np.ndarray] = None,
    ) -> Tuple[float, float, np.ndarray]:
        """
        Estimate ATT using outcome regression.

        With covariates:
        1. Regress outcome changes on covariates for control group
        2. Predict counterfactual for treated using their covariates
        3. ATT = mean(treated_change) - mean(predicted_counterfactual)

        Without covariates:
        Simple difference in means.
        """
        n_t = len(treated_change)
        n_c = len(control_change)

        if X_treated is not None and X_control is not None and X_treated.shape[1] > 0:
            # Covariate-adjusted outcome regression
            # Fit regression on control units: E[Delta Y | X, D=0]
            beta, residuals = _linear_regression(X_control, control_change)

            # Predict counterfactual for treated units
            X_treated_with_intercept = np.column_stack([np.ones(n_t), X_treated])
            predicted_control = X_treated_with_intercept @ beta

            # ATT = mean(observed treated change - predicted counterfactual)
            att = np.mean(treated_change - predicted_control)

            # Standard error using sandwich estimator
            # Variance from treated: Var(Y_1 - m(X))
            treated_residuals = treated_change - predicted_control
            var_t = np.var(treated_residuals, ddof=1) if n_t > 1 else 0.0

            # Variance from control regression (residual variance)
            var_c = np.var(residuals, ddof=1) if n_c > 1 else 0.0

            # Approximate SE (ignoring estimation error in beta for simplicity)
            se = np.sqrt(var_t / n_t + var_c / n_c) if (n_t > 0 and n_c > 0) else 0.0

            # Influence function
            inf_treated = (treated_residuals - np.mean(treated_residuals)) / n_t
            inf_control = -residuals / n_c
            inf_func = np.concatenate([inf_treated, inf_control])
        else:
            # Simple difference in means (no covariates)
            att = np.mean(treated_change) - np.mean(control_change)

            var_t = np.var(treated_change, ddof=1) if n_t > 1 else 0.0
            var_c = np.var(control_change, ddof=1) if n_c > 1 else 0.0

            se = np.sqrt(var_t / n_t + var_c / n_c) if (n_t > 0 and n_c > 0) else 0.0

            # Influence function (for aggregation)
            inf_treated = treated_change - np.mean(treated_change)
            inf_control = control_change - np.mean(control_change)
            inf_func = np.concatenate([inf_treated / n_t, -inf_control / n_c])

        return att, se, inf_func

    def _ipw_estimation(
        self,
        treated_change: np.ndarray,
        control_change: np.ndarray,
        n_treated: int,
        n_control: int,
        X_treated: Optional[np.ndarray] = None,
        X_control: Optional[np.ndarray] = None,
    ) -> Tuple[float, float, np.ndarray]:
        """
        Estimate ATT using inverse probability weighting.

        With covariates:
        1. Estimate propensity score P(D=1|X) using logistic regression
        2. Reweight control units to match treated covariate distribution
        3. ATT = mean(treated) - weighted_mean(control)

        Without covariates:
        Simple difference in means with unconditional propensity weighting.
        """
        n_t = len(treated_change)
        n_c = len(control_change)
        n_total = n_treated + n_control

        if X_treated is not None and X_control is not None and X_treated.shape[1] > 0:
            # Covariate-adjusted IPW estimation
            # Stack covariates and create treatment indicator
            X_all = np.vstack([X_treated, X_control])
            D = np.concatenate([np.ones(n_t), np.zeros(n_c)])

            # Estimate propensity scores using logistic regression
            try:
                _, pscore = _logistic_regression(X_all, D)
            except (np.linalg.LinAlgError, ValueError):
                # Fallback to unconditional if logistic regression fails
                warnings.warn(
                    "Propensity score estimation failed. "
                    "Falling back to unconditional estimation.",
                    UserWarning,
                    stacklevel=4,
                )
                pscore = np.full(len(D), n_t / (n_t + n_c))

            # Propensity scores for treated and control
            pscore_treated = pscore[:n_t]
            pscore_control = pscore[n_t:]

            # Clip propensity scores to avoid extreme weights
            pscore_control = np.clip(pscore_control, 0.01, 0.99)
            pscore_treated = np.clip(pscore_treated, 0.01, 0.99)

            # IPW weights for control units: p(X) / (1 - p(X))
            # This reweights controls to have same covariate distribution as treated
            weights_control = pscore_control / (1 - pscore_control)
            weights_control = weights_control / np.sum(weights_control)  # normalize

            # ATT = mean(treated) - weighted_mean(control)
            att = np.mean(treated_change) - np.sum(weights_control * control_change)

            # Compute standard error
            # Variance of treated mean
            var_t = np.var(treated_change, ddof=1) if n_t > 1 else 0.0

            # Variance of weighted control mean
            weighted_var_c = np.sum(weights_control * (control_change - np.sum(weights_control * control_change)) ** 2)

            se = np.sqrt(var_t / n_t + weighted_var_c) if (n_t > 0 and n_c > 0) else 0.0

            # Influence function
            inf_treated = (treated_change - np.mean(treated_change)) / n_t
            inf_control = -weights_control * (control_change - np.sum(weights_control * control_change))
            inf_func = np.concatenate([inf_treated, inf_control])
        else:
            # Unconditional IPW (reduces to difference in means)
            p_treat = n_treated / n_total  # unconditional propensity score

            att = np.mean(treated_change) - np.mean(control_change)

            var_t = np.var(treated_change, ddof=1) if n_t > 1 else 0.0
            var_c = np.var(control_change, ddof=1) if n_c > 1 else 0.0

            # Adjusted variance for IPW
            se = np.sqrt(var_t / n_t + var_c * (1 - p_treat) / (n_c * p_treat)) if (n_t > 0 and n_c > 0 and p_treat > 0) else 0.0

            # Influence function (for aggregation)
            inf_treated = (treated_change - np.mean(treated_change)) / n_t
            inf_control = (control_change - np.mean(control_change)) / n_c
            inf_func = np.concatenate([inf_treated, -inf_control])

        return att, se, inf_func

    def _doubly_robust(
        self,
        treated_change: np.ndarray,
        control_change: np.ndarray,
        X_treated: Optional[np.ndarray] = None,
        X_control: Optional[np.ndarray] = None,
    ) -> Tuple[float, float, np.ndarray]:
        """
        Estimate ATT using doubly robust estimation.

        With covariates:
        Combines outcome regression and IPW for double robustness.
        The estimator is consistent if either the outcome model OR
        the propensity model is correctly specified.

        ATT_DR = (1/n_t) * sum_i[D_i * (Y_i - m(X_i))]
               + (1/n_t) * sum_i[(1-D_i) * w_i * (m(X_i) - Y_i)]

        where m(X) is the outcome model and w_i are IPW weights.

        Without covariates:
        Reduces to simple difference in means.
        """
        n_t = len(treated_change)
        n_c = len(control_change)

        if X_treated is not None and X_control is not None and X_treated.shape[1] > 0:
            # Doubly robust estimation with covariates
            # Step 1: Outcome regression - fit E[Delta Y | X] on control
            beta, _ = _linear_regression(X_control, control_change)

            # Predict counterfactual for both treated and control
            X_treated_with_intercept = np.column_stack([np.ones(n_t), X_treated])
            X_control_with_intercept = np.column_stack([np.ones(n_c), X_control])
            m_treated = X_treated_with_intercept @ beta
            m_control = X_control_with_intercept @ beta

            # Step 2: Propensity score estimation
            X_all = np.vstack([X_treated, X_control])
            D = np.concatenate([np.ones(n_t), np.zeros(n_c)])

            try:
                _, pscore = _logistic_regression(X_all, D)
            except (np.linalg.LinAlgError, ValueError):
                # Fallback to unconditional if logistic regression fails
                pscore = np.full(len(D), n_t / (n_t + n_c))

            pscore_control = pscore[n_t:]

            # Clip propensity scores
            pscore_control = np.clip(pscore_control, 0.01, 0.99)

            # IPW weights for control: p(X) / (1 - p(X))
            weights_control = pscore_control / (1 - pscore_control)

            # Step 3: Doubly robust ATT
            # ATT = mean(treated - m(X_treated))
            #     + weighted_mean_control((m(X) - Y) * weight)
            att_treated_part = np.mean(treated_change - m_treated)

            # Augmentation term from control
            augmentation = np.sum(weights_control * (m_control - control_change)) / n_t

            att = att_treated_part + augmentation

            # Step 4: Standard error using influence function
            # Influence function for DR estimator
            psi_treated = (treated_change - m_treated - att) / n_t
            psi_control = (weights_control * (m_control - control_change)) / n_t

            # Variance is sum of squared influence functions
            var_psi = np.sum(psi_treated ** 2) + np.sum(psi_control ** 2)
            se = np.sqrt(var_psi) if var_psi > 0 else 0.0

            # Full influence function
            inf_func = np.concatenate([psi_treated, psi_control])
        else:
            # Without covariates, DR simplifies to difference in means
            att = np.mean(treated_change) - np.mean(control_change)

            var_t = np.var(treated_change, ddof=1) if n_t > 1 else 0.0
            var_c = np.var(control_change, ddof=1) if n_c > 1 else 0.0

            se = np.sqrt(var_t / n_t + var_c / n_c) if (n_t > 0 and n_c > 0) else 0.0

            # Influence function for DR estimator
            inf_treated = (treated_change - np.mean(treated_change)) / n_t
            inf_control = (control_change - np.mean(control_change)) / n_c
            inf_func = np.concatenate([inf_treated, -inf_control])

        return att, se, inf_func

    def _aggregate_simple(
        self,
        group_time_effects: Dict,
        df: pd.DataFrame,
        unit: str,
    ) -> Tuple[float, float]:
        """
        Compute simple weighted average of ATT(g,t).

        Weights by group size (number of treated units).
        """
        effects = []
        weights = []
        variances = []

        for (g, t), data in group_time_effects.items():
            effects.append(data['effect'])
            weights.append(data['n_treated'])
            variances.append(data['se'] ** 2)

        effects = np.array(effects)
        weights = np.array(weights, dtype=float)
        variances = np.array(variances)

        # Normalize weights
        weights = weights / np.sum(weights)

        # Weighted average
        overall_att = np.sum(weights * effects)

        # Standard error (assuming independence across g,t)
        overall_var = np.sum((weights ** 2) * variances)
        overall_se = np.sqrt(overall_var)

        return overall_att, overall_se

    def _aggregate_event_study(
        self,
        group_time_effects: Dict,
        groups: List[Any],
        time_periods: List[Any],
        balance_e: Optional[int] = None,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Aggregate effects by relative time (event study).

        Computes average effect at each event time e = t - g.
        """
        # Organize effects by relative time
        effects_by_e: Dict[int, List[Tuple[float, float, int]]] = {}

        for (g, t), data in group_time_effects.items():
            e = t - g  # Relative time
            if e not in effects_by_e:
                effects_by_e[e] = []
            effects_by_e[e].append((
                data['effect'],
                data['se'],
                data['n_treated']
            ))

        # Balance the panel if requested
        if balance_e is not None:
            # Keep only groups that have effects at relative time balance_e
            groups_at_e = set()
            for (g, t), data in group_time_effects.items():
                if t - g == balance_e:
                    groups_at_e.add(g)

            # Filter effects to only include balanced groups
            balanced_effects: Dict[int, List[Tuple[float, float, int]]] = {}
            for (g, t), data in group_time_effects.items():
                if g in groups_at_e:
                    e = t - g
                    if e not in balanced_effects:
                        balanced_effects[e] = []
                    balanced_effects[e].append((
                        data['effect'],
                        data['se'],
                        data['n_treated']
                    ))
            effects_by_e = balanced_effects

        # Compute aggregated effects
        event_study_effects = {}

        for e, effect_list in sorted(effects_by_e.items()):
            effs = np.array([x[0] for x in effect_list])
            ses = np.array([x[1] for x in effect_list])
            ns = np.array([x[2] for x in effect_list], dtype=float)

            # Weight by group size
            weights = ns / np.sum(ns)

            agg_effect = np.sum(weights * effs)
            agg_var = np.sum((weights ** 2) * (ses ** 2))
            agg_se = np.sqrt(agg_var)

            t_stat = agg_effect / agg_se if agg_se > 0 else 0.0
            p_val = compute_p_value(t_stat)
            ci = compute_confidence_interval(agg_effect, agg_se, self.alpha)

            event_study_effects[e] = {
                'effect': agg_effect,
                'se': agg_se,
                't_stat': t_stat,
                'p_value': p_val,
                'conf_int': ci,
                'n_groups': len(effect_list),
            }

        return event_study_effects

    def _aggregate_by_group(
        self,
        group_time_effects: Dict,
        groups: List[Any],
    ) -> Dict[Any, Dict[str, Any]]:
        """
        Aggregate effects by treatment cohort.

        Computes average effect for each cohort across all post-treatment periods.
        """
        group_effects = {}

        for g in groups:
            # Get all effects for this group (post-treatment only: t >= g)
            g_effects = [
                (data['effect'], data['se'], data['n_treated'])
                for (gg, t), data in group_time_effects.items()
                if gg == g and t >= g
            ]

            if not g_effects:
                continue

            effs = np.array([x[0] for x in g_effects])
            ses = np.array([x[1] for x in g_effects])

            # Equal weight across time periods for a group
            weights = np.ones(len(effs)) / len(effs)

            agg_effect = np.sum(weights * effs)
            agg_var = np.sum((weights ** 2) * (ses ** 2))
            agg_se = np.sqrt(agg_var)

            t_stat = agg_effect / agg_se if agg_se > 0 else 0.0
            p_val = compute_p_value(t_stat)
            ci = compute_confidence_interval(agg_effect, agg_se, self.alpha)

            group_effects[g] = {
                'effect': agg_effect,
                'se': agg_se,
                't_stat': t_stat,
                'p_value': p_val,
                'conf_int': ci,
                'n_periods': len(g_effects),
            }

        return group_effects

    def _run_multiplier_bootstrap(
        self,
        group_time_effects: Dict[Tuple[Any, Any], Dict[str, Any]],
        influence_func_info: Dict[Tuple[Any, Any], Dict[str, Any]],
        aggregate: Optional[str],
        balance_e: Optional[int],
        treatment_groups: List[Any],
        time_periods: List[Any],
    ) -> CSBootstrapResults:
        """
        Run multiplier bootstrap for inference on all parameters.

        This implements the multiplier bootstrap procedure from Callaway & Sant'Anna (2021).
        The key idea is to perturb the influence function contributions with random
        weights at the cluster (unit) level, then recompute aggregations.

        Parameters
        ----------
        group_time_effects : dict
            Dictionary of ATT(g,t) effects with analytical SEs.
        influence_func_info : dict
            Dictionary mapping (g,t) to influence function information.
        aggregate : str, optional
            Type of aggregation requested.
        balance_e : int, optional
            Balance parameter for event study.
        treatment_groups : list
            List of treatment cohorts.
        time_periods : list
            List of time periods.

        Returns
        -------
        CSBootstrapResults
            Bootstrap inference results.
        """
        # Warn about low bootstrap iterations
        if self.n_bootstrap < 50:
            warnings.warn(
                f"n_bootstrap={self.n_bootstrap} is low. Consider n_bootstrap >= 199 "
                "for reliable inference. Percentile confidence intervals and p-values "
                "may be unreliable with few iterations.",
                UserWarning,
                stacklevel=3,
            )

        rng = np.random.default_rng(self.seed)

        # Collect all unique units across all (g,t) combinations
        all_units = set()
        for (g, t), info in influence_func_info.items():
            all_units.update(info['treated_units'])
            all_units.update(info['control_units'])
        all_units = sorted(all_units)
        n_units = len(all_units)
        unit_to_idx = {u: i for i, u in enumerate(all_units)}

        # Get list of (g,t) pairs
        gt_pairs = list(group_time_effects.keys())
        n_gt = len(gt_pairs)

        # Compute aggregation weights for overall ATT
        overall_weights = np.array([
            group_time_effects[gt]['n_treated'] for gt in gt_pairs
        ], dtype=float)
        overall_weights = overall_weights / np.sum(overall_weights)

        # Original point estimates
        original_atts = np.array([group_time_effects[gt]['effect'] for gt in gt_pairs])
        original_overall = np.sum(overall_weights * original_atts)

        # Prepare event study and group aggregation info if needed
        event_study_info = None
        group_agg_info = None

        if aggregate in ["event_study", "all"]:
            event_study_info = self._prepare_event_study_aggregation(
                gt_pairs, group_time_effects, balance_e
            )

        if aggregate in ["group", "all"]:
            group_agg_info = self._prepare_group_aggregation(
                gt_pairs, group_time_effects, treatment_groups
            )

        # Bootstrap arrays to store results
        bootstrap_atts_gt = np.zeros((self.n_bootstrap, n_gt))
        bootstrap_overall = np.zeros(self.n_bootstrap)

        if event_study_info is not None:
            rel_periods = sorted(event_study_info.keys())
            bootstrap_event_study = {e: np.zeros(self.n_bootstrap) for e in rel_periods}
        else:
            bootstrap_event_study = None

        if group_agg_info is not None:
            groups = sorted(group_agg_info.keys())
            bootstrap_group = {g: np.zeros(self.n_bootstrap) for g in groups}
        else:
            bootstrap_group = None

        # Run bootstrap iterations
        for b in range(self.n_bootstrap):
            # Generate unit-level weights
            unit_weights = _generate_bootstrap_weights(
                n_units, self.bootstrap_weight_type, rng
            )

            # Compute bootstrap ATT(g,t) for each group-time pair
            for j, gt in enumerate(gt_pairs):
                info = influence_func_info[gt]

                # Get weights for treated and control units
                treated_indices = [unit_to_idx[u] for u in info['treated_units']]
                control_indices = [unit_to_idx[u] for u in info['control_units']]

                treated_weights = unit_weights[treated_indices]
                control_weights = unit_weights[control_indices]

                # Influence function perturbation
                # Bootstrap ATT* = ATT + sum(weights * influence)
                perturbation = (
                    np.sum(treated_weights * info['treated_inf']) +
                    np.sum(control_weights * info['control_inf'])
                )

                bootstrap_atts_gt[b, j] = original_atts[j] + perturbation

            # Compute bootstrap overall ATT
            bootstrap_overall[b] = np.sum(overall_weights * bootstrap_atts_gt[b, :])

            # Compute bootstrap event study effects
            if bootstrap_event_study is not None and event_study_info is not None:
                for e, agg_info in event_study_info.items():
                    gt_indices = agg_info['gt_indices']
                    weights = agg_info['weights']
                    bootstrap_event_study[e][b] = np.sum(
                        weights * bootstrap_atts_gt[b, gt_indices]
                    )

            # Compute bootstrap group effects
            if bootstrap_group is not None and group_agg_info is not None:
                for g, agg_info in group_agg_info.items():
                    gt_indices = agg_info['gt_indices']
                    weights = agg_info['weights']
                    bootstrap_group[g][b] = np.sum(
                        weights * bootstrap_atts_gt[b, gt_indices]
                    )

        # Compute bootstrap statistics
        # ATT(g,t) statistics
        gt_ses = {}
        gt_cis = {}
        gt_p_values = {}

        for j, gt in enumerate(gt_pairs):
            original_effect = original_atts[j]
            boot_dist = bootstrap_atts_gt[:, j]

            se = float(np.std(boot_dist, ddof=1))
            ci = self._compute_percentile_ci(boot_dist, self.alpha)
            p_value = self._compute_bootstrap_pvalue(original_effect, boot_dist)

            gt_ses[gt] = se
            gt_cis[gt] = ci
            gt_p_values[gt] = p_value

        # Overall ATT statistics
        overall_se = float(np.std(bootstrap_overall, ddof=1))
        overall_ci = self._compute_percentile_ci(bootstrap_overall, self.alpha)
        overall_p_value = self._compute_bootstrap_pvalue(original_overall, bootstrap_overall)

        # Event study statistics
        event_study_ses = None
        event_study_cis = None
        event_study_p_values = None

        if bootstrap_event_study is not None and event_study_info is not None:
            event_study_ses = {}
            event_study_cis = {}
            event_study_p_values = {}

            for e in rel_periods:
                original_effect = event_study_info[e]['effect']
                boot_dist = bootstrap_event_study[e]

                event_study_ses[e] = float(np.std(boot_dist, ddof=1))
                event_study_cis[e] = self._compute_percentile_ci(boot_dist, self.alpha)
                event_study_p_values[e] = self._compute_bootstrap_pvalue(
                    original_effect, boot_dist
                )

        # Group effect statistics
        group_effect_ses = None
        group_effect_cis = None
        group_effect_p_values = None

        if bootstrap_group is not None and group_agg_info is not None:
            group_effect_ses = {}
            group_effect_cis = {}
            group_effect_p_values = {}

            for g in groups:
                original_effect = group_agg_info[g]['effect']
                boot_dist = bootstrap_group[g]

                group_effect_ses[g] = float(np.std(boot_dist, ddof=1))
                group_effect_cis[g] = self._compute_percentile_ci(boot_dist, self.alpha)
                group_effect_p_values[g] = self._compute_bootstrap_pvalue(
                    original_effect, boot_dist
                )

        return CSBootstrapResults(
            n_bootstrap=self.n_bootstrap,
            weight_type=self.bootstrap_weight_type,
            alpha=self.alpha,
            overall_att_se=overall_se,
            overall_att_ci=overall_ci,
            overall_att_p_value=overall_p_value,
            group_time_ses=gt_ses,
            group_time_cis=gt_cis,
            group_time_p_values=gt_p_values,
            event_study_ses=event_study_ses,
            event_study_cis=event_study_cis,
            event_study_p_values=event_study_p_values,
            group_effect_ses=group_effect_ses,
            group_effect_cis=group_effect_cis,
            group_effect_p_values=group_effect_p_values,
            bootstrap_distribution=bootstrap_overall,
        )

    def _prepare_event_study_aggregation(
        self,
        gt_pairs: List[Tuple[Any, Any]],
        group_time_effects: Dict,
        balance_e: Optional[int],
    ) -> Dict[int, Dict[str, Any]]:
        """Prepare aggregation info for event study bootstrap."""
        # Organize by relative time
        effects_by_e: Dict[int, List[Tuple[int, float, float]]] = {}

        for j, (g, t) in enumerate(gt_pairs):
            e = t - g
            if e not in effects_by_e:
                effects_by_e[e] = []
            effects_by_e[e].append((
                j,  # index in gt_pairs
                group_time_effects[(g, t)]['effect'],
                group_time_effects[(g, t)]['n_treated']
            ))

        # Balance if requested
        if balance_e is not None:
            groups_at_e = set()
            for j, (g, t) in enumerate(gt_pairs):
                if t - g == balance_e:
                    groups_at_e.add(g)

            balanced_effects: Dict[int, List[Tuple[int, float, float]]] = {}
            for j, (g, t) in enumerate(gt_pairs):
                if g in groups_at_e:
                    e = t - g
                    if e not in balanced_effects:
                        balanced_effects[e] = []
                    balanced_effects[e].append((
                        j,
                        group_time_effects[(g, t)]['effect'],
                        group_time_effects[(g, t)]['n_treated']
                    ))
            effects_by_e = balanced_effects

        # Compute aggregation weights
        result = {}
        for e, effect_list in effects_by_e.items():
            indices = np.array([x[0] for x in effect_list])
            effects = np.array([x[1] for x in effect_list])
            n_treated = np.array([x[2] for x in effect_list], dtype=float)

            weights = n_treated / np.sum(n_treated)
            agg_effect = np.sum(weights * effects)

            result[e] = {
                'gt_indices': indices,
                'weights': weights,
                'effect': agg_effect,
            }

        return result

    def _prepare_group_aggregation(
        self,
        gt_pairs: List[Tuple[Any, Any]],
        group_time_effects: Dict,
        treatment_groups: List[Any],
    ) -> Dict[Any, Dict[str, Any]]:
        """Prepare aggregation info for group-level bootstrap."""
        result = {}

        for g in treatment_groups:
            # Get all effects for this group (post-treatment only: t >= g)
            group_data = []
            for j, (gg, t) in enumerate(gt_pairs):
                if gg == g and t >= g:
                    group_data.append((
                        j,
                        group_time_effects[(gg, t)]['effect'],
                    ))

            if not group_data:
                continue

            indices = np.array([x[0] for x in group_data])
            effects = np.array([x[1] for x in group_data])

            # Equal weights across time periods
            weights = np.ones(len(effects)) / len(effects)
            agg_effect = np.sum(weights * effects)

            result[g] = {
                'gt_indices': indices,
                'weights': weights,
                'effect': agg_effect,
            }

        return result

    def _compute_percentile_ci(
        self,
        boot_dist: np.ndarray,
        alpha: float,
    ) -> Tuple[float, float]:
        """Compute percentile confidence interval from bootstrap distribution."""
        lower = float(np.percentile(boot_dist, alpha / 2 * 100))
        upper = float(np.percentile(boot_dist, (1 - alpha / 2) * 100))
        return (lower, upper)

    def _compute_bootstrap_pvalue(
        self,
        original_effect: float,
        boot_dist: np.ndarray,
    ) -> float:
        """
        Compute two-sided bootstrap p-value.

        Uses the percentile method: p-value is the proportion of bootstrap
        estimates on the opposite side of zero from the original estimate,
        doubled for two-sided test.
        """
        if original_effect >= 0:
            # Proportion of bootstrap estimates <= 0
            p_one_sided = np.mean(boot_dist <= 0)
        else:
            # Proportion of bootstrap estimates >= 0
            p_one_sided = np.mean(boot_dist >= 0)

        # Two-sided p-value
        p_value = min(2 * p_one_sided, 1.0)

        # Ensure minimum p-value
        p_value = max(p_value, 1 / (self.n_bootstrap + 1))

        return float(p_value)

    def get_params(self) -> Dict[str, Any]:
        """Get estimator parameters (sklearn-compatible)."""
        return {
            "control_group": self.control_group,
            "anticipation": self.anticipation,
            "estimation_method": self.estimation_method,
            "alpha": self.alpha,
            "cluster": self.cluster,
            "n_bootstrap": self.n_bootstrap,
            "bootstrap_weights": self.bootstrap_weights,
            # Deprecated but kept for backward compatibility
            "bootstrap_weight_type": self.bootstrap_weight_type,
            "seed": self.seed,
        }

    def set_params(self, **params) -> "CallawaySantAnna":
        """Set estimator parameters (sklearn-compatible)."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")
        return self

    def summary(self) -> str:
        """Get summary of estimation results."""
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before calling summary()")
        assert self.results_ is not None
        return self.results_.summary()

    def print_summary(self) -> None:
        """Print summary to stdout."""
        print(self.summary())
