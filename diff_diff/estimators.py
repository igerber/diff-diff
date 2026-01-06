"""
Difference-in-Differences estimators with sklearn-like API.

This module contains the core DiD estimators:
- DifferenceInDifferences: Basic 2x2 DiD estimator
- MultiPeriodDiD: Event-study style DiD with period-specific treatment effects

Additional estimators are in separate modules:
- TwoWayFixedEffects: See diff_diff.twfe
- SyntheticDiD: See diff_diff.synthetic_did

For backward compatibility, all estimators are re-exported from this module.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from diff_diff.results import DiDResults, MultiPeriodDiDResults, PeriodEffect
from diff_diff.utils import (
    WildBootstrapResults,
    compute_confidence_interval,
    compute_p_value,
    compute_robust_se,
    validate_binary,
    wild_bootstrap_se,
)


class DifferenceInDifferences:
    """
    Difference-in-Differences estimator with sklearn-like interface.

    Estimates the Average Treatment effect on the Treated (ATT) using
    the canonical 2x2 DiD design or panel data with two-way fixed effects.

    Parameters
    ----------
    formula : str, optional
        R-style formula for the model (e.g., "outcome ~ treated * post").
        If provided, overrides column name parameters.
    robust : bool, default=True
        Whether to use heteroskedasticity-robust standard errors (HC1).
    cluster : str, optional
        Column name for cluster-robust standard errors.
    alpha : float, default=0.05
        Significance level for confidence intervals.
    inference : str, default="analytical"
        Inference method: "analytical" for standard asymptotic inference,
        or "wild_bootstrap" for wild cluster bootstrap (recommended when
        number of clusters is small, <50).
    n_bootstrap : int, default=999
        Number of bootstrap replications when inference="wild_bootstrap".
    bootstrap_weights : str, default="rademacher"
        Type of bootstrap weights: "rademacher" (standard), "webb"
        (recommended for <10 clusters), or "mammen" (skewness correction).
    seed : int, optional
        Random seed for reproducibility when using bootstrap inference.
        If None (default), results will vary between runs.

    Attributes
    ----------
    results_ : DiDResults
        Estimation results after calling fit().
    is_fitted_ : bool
        Whether the model has been fitted.

    Examples
    --------
    Basic usage with a DataFrame:

    >>> import pandas as pd
    >>> from diff_diff import DifferenceInDifferences
    >>>
    >>> # Create sample data
    >>> data = pd.DataFrame({
    ...     'outcome': [10, 11, 15, 18, 9, 10, 12, 13],
    ...     'treated': [1, 1, 1, 1, 0, 0, 0, 0],
    ...     'post': [0, 0, 1, 1, 0, 0, 1, 1]
    ... })
    >>>
    >>> # Fit the model
    >>> did = DifferenceInDifferences()
    >>> results = did.fit(data, outcome='outcome', treatment='treated', time='post')
    >>>
    >>> # View results
    >>> print(results.att)  # ATT estimate
    >>> results.print_summary()  # Full summary table

    Using formula interface:

    >>> did = DifferenceInDifferences()
    >>> results = did.fit(data, formula='outcome ~ treated * post')

    Notes
    -----
    The ATT is computed using the standard DiD formula:

        ATT = (E[Y|D=1,T=1] - E[Y|D=1,T=0]) - (E[Y|D=0,T=1] - E[Y|D=0,T=0])

    Or equivalently via OLS regression:

        Y = α + β₁*D + β₂*T + β₃*(D×T) + ε

    Where β₃ is the ATT.
    """

    def __init__(
        self,
        robust: bool = True,
        cluster: Optional[str] = None,
        alpha: float = 0.05,
        inference: str = "analytical",
        n_bootstrap: int = 999,
        bootstrap_weights: str = "rademacher",
        seed: Optional[int] = None
    ):
        self.robust = robust
        self.cluster = cluster
        self.alpha = alpha
        self.inference = inference
        self.n_bootstrap = n_bootstrap
        self.bootstrap_weights = bootstrap_weights
        self.seed = seed

        self.is_fitted_ = False
        self.results_ = None
        self._coefficients = None
        self._vcov = None
        self._bootstrap_results = None  # Store WildBootstrapResults if used

    def fit(
        self,
        data: pd.DataFrame,
        outcome: Optional[str] = None,
        treatment: Optional[str] = None,
        time: Optional[str] = None,
        formula: Optional[str] = None,
        covariates: Optional[List[str]] = None,
        fixed_effects: Optional[List[str]] = None,
        absorb: Optional[List[str]] = None
    ) -> DiDResults:
        """
        Fit the Difference-in-Differences model.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing the outcome, treatment, and time variables.
        outcome : str
            Name of the outcome variable column.
        treatment : str
            Name of the treatment group indicator column (0/1).
        time : str
            Name of the post-treatment period indicator column (0/1).
        formula : str, optional
            R-style formula (e.g., "outcome ~ treated * post").
            If provided, overrides outcome, treatment, and time parameters.
        covariates : list, optional
            List of covariate column names to include as linear controls.
        fixed_effects : list, optional
            List of categorical column names to include as fixed effects.
            Creates dummy variables for each category (drops first level).
            Use for low-dimensional fixed effects (e.g., industry, region).
        absorb : list, optional
            List of categorical column names for high-dimensional fixed effects.
            Uses within-transformation (demeaning) instead of dummy variables.
            More efficient for large numbers of categories (e.g., firm, individual).

        Returns
        -------
        DiDResults
            Object containing estimation results.

        Raises
        ------
        ValueError
            If required parameters are missing or data validation fails.

        Examples
        --------
        Using fixed effects (dummy variables):

        >>> did.fit(data, outcome='sales', treatment='treated', time='post',
        ...         fixed_effects=['state', 'industry'])

        Using absorbed fixed effects (within-transformation):

        >>> did.fit(data, outcome='sales', treatment='treated', time='post',
        ...         absorb=['firm_id'])
        """
        # Parse formula if provided
        if formula is not None:
            outcome, treatment, time, covariates = self._parse_formula(formula, data)
        elif outcome is None or treatment is None or time is None:
            raise ValueError(
                "Must provide either 'formula' or all of 'outcome', 'treatment', and 'time'"
            )

        # Validate inputs
        self._validate_data(data, outcome, treatment, time, covariates)

        # Validate binary variables BEFORE any transformations
        validate_binary(data[treatment].values, "treatment")
        validate_binary(data[time].values, "time")

        # Validate fixed effects and absorb columns
        if fixed_effects:
            for fe in fixed_effects:
                if fe not in data.columns:
                    raise ValueError(f"Fixed effect column '{fe}' not found in data")
        if absorb:
            for ab in absorb:
                if ab not in data.columns:
                    raise ValueError(f"Absorb column '{ab}' not found in data")

        # Handle absorbed fixed effects (within-transformation)
        working_data = data.copy()
        absorbed_vars = []
        n_absorbed_effects = 0

        if absorb:
            # Apply within-transformation for each absorbed variable
            # Only demean outcome and covariates, NOT treatment/time indicators
            # Treatment is typically time-invariant (within unit), and time is
            # unit-invariant, so demeaning them would create multicollinearity
            vars_to_demean = [outcome] + (covariates or [])
            for ab_var in absorb:
                n_absorbed_effects += working_data[ab_var].nunique() - 1
                for var in vars_to_demean:
                    group_means = working_data.groupby(ab_var)[var].transform("mean")
                    working_data[var] = working_data[var] - group_means
                absorbed_vars.append(ab_var)

        # Extract variables (may be demeaned if absorb was used)
        y = working_data[outcome].values.astype(float)
        d = working_data[treatment].values.astype(float)
        t = working_data[time].values.astype(float)

        # Create interaction term
        dt = d * t

        # Build design matrix
        X = np.column_stack([np.ones(len(y)), d, t, dt])
        var_names = ["const", treatment, time, f"{treatment}:{time}"]

        # Add covariates if provided
        if covariates:
            for cov in covariates:
                X = np.column_stack([X, working_data[cov].values.astype(float)])
                var_names.append(cov)

        # Add fixed effects as dummy variables
        if fixed_effects:
            for fe in fixed_effects:
                # Create dummies, drop first category to avoid multicollinearity
                # Use working_data to be consistent with absorbed FE if both are used
                dummies = pd.get_dummies(working_data[fe], prefix=fe, drop_first=True)
                for col in dummies.columns:
                    X = np.column_stack([X, dummies[col].values.astype(float)])
                    var_names.append(col)

        # Fit OLS
        coefficients, residuals, fitted, r_squared = self._fit_ols(X, y)

        # Extract ATT (coefficient on interaction term)
        att_idx = 3  # Index of interaction term
        att_var_name = f"{treatment}:{time}"
        assert var_names[att_idx] == att_var_name, (
            f"ATT index mismatch: expected '{att_var_name}' at index {att_idx}, "
            f"but found '{var_names[att_idx]}'"
        )
        att = coefficients[att_idx]

        # Compute degrees of freedom (used for analytical inference)
        df = len(y) - X.shape[1] - n_absorbed_effects

        # Compute standard errors and inference
        if self.inference == "wild_bootstrap" and self.cluster is not None:
            # Wild cluster bootstrap for few-cluster inference
            cluster_ids = data[self.cluster].values
            se, p_value, conf_int, t_stat, vcov, _ = self._run_wild_bootstrap_inference(
                X, y, residuals, cluster_ids, att_idx
            )
        elif self.cluster is not None:
            cluster_ids = data[self.cluster].values
            vcov = compute_robust_se(X, residuals, cluster_ids)
            se = np.sqrt(vcov[att_idx, att_idx])
            t_stat = att / se
            p_value = compute_p_value(t_stat, df=df)
            conf_int = compute_confidence_interval(att, se, self.alpha, df=df)
        elif self.robust:
            vcov = compute_robust_se(X, residuals)
            se = np.sqrt(vcov[att_idx, att_idx])
            t_stat = att / se
            p_value = compute_p_value(t_stat, df=df)
            conf_int = compute_confidence_interval(att, se, self.alpha, df=df)
        else:
            # Classical OLS standard errors
            n = len(y)
            k = X.shape[1]
            mse = np.sum(residuals ** 2) / (n - k)
            # Use solve() instead of inv() for numerical stability
            # solve(A, B) computes X where AX=B, so this yields (X'X)^{-1} * mse
            vcov = np.linalg.solve(X.T @ X, mse * np.eye(k))
            se = np.sqrt(vcov[att_idx, att_idx])
            t_stat = att / se
            p_value = compute_p_value(t_stat, df=df)
            conf_int = compute_confidence_interval(att, se, self.alpha, df=df)

        # Count observations
        n_treated = int(np.sum(d))
        n_control = int(np.sum(1 - d))

        # Create coefficient dictionary
        coef_dict = {name: coef for name, coef in zip(var_names, coefficients)}

        # Determine inference method and bootstrap info
        inference_method = "analytical"
        n_bootstrap_used = None
        n_clusters_used = None
        if self._bootstrap_results is not None:
            inference_method = "wild_bootstrap"
            n_bootstrap_used = self._bootstrap_results.n_bootstrap
            n_clusters_used = self._bootstrap_results.n_clusters

        # Store results
        self.results_ = DiDResults(
            att=att,
            se=se,
            t_stat=t_stat,
            p_value=p_value,
            conf_int=conf_int,
            n_obs=len(y),
            n_treated=n_treated,
            n_control=n_control,
            alpha=self.alpha,
            coefficients=coef_dict,
            vcov=vcov,
            residuals=residuals,
            fitted_values=fitted,
            r_squared=r_squared,
            inference_method=inference_method,
            n_bootstrap=n_bootstrap_used,
            n_clusters=n_clusters_used,
        )

        self._coefficients = coefficients
        self._vcov = vcov
        self.is_fitted_ = True

        return self.results_

    def _fit_ols(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Fit OLS regression.

        Parameters
        ----------
        X : np.ndarray
            Design matrix.
        y : np.ndarray
            Outcome vector.

        Returns
        -------
        tuple
            (coefficients, residuals, fitted_values, r_squared)

        Raises
        ------
        ValueError
            If design matrix is rank-deficient (perfect multicollinearity).
        """
        # Check for rank deficiency (perfect multicollinearity)
        rank = np.linalg.matrix_rank(X)
        if rank < X.shape[1]:
            raise ValueError(
                f"Design matrix is rank-deficient (rank {rank} < {X.shape[1]} columns). "
                "This indicates perfect multicollinearity. Check your fixed effects "
                "and covariates for linear dependencies."
            )

        # Solve normal equations: β = (X'X)^(-1) X'y
        coefficients = np.linalg.lstsq(X, y, rcond=None)[0]

        # Compute fitted values and residuals
        fitted = X @ coefficients
        residuals = y - fitted

        # Compute R-squared
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return coefficients, residuals, fitted, r_squared

    def _run_wild_bootstrap_inference(
        self,
        X: np.ndarray,
        y: np.ndarray,
        residuals: np.ndarray,
        cluster_ids: np.ndarray,
        coefficient_index: int,
    ) -> Tuple[float, float, Tuple[float, float], float, np.ndarray, WildBootstrapResults]:
        """
        Run wild cluster bootstrap inference.

        Parameters
        ----------
        X : np.ndarray
            Design matrix.
        y : np.ndarray
            Outcome vector.
        residuals : np.ndarray
            OLS residuals.
        cluster_ids : np.ndarray
            Cluster identifiers for each observation.
        coefficient_index : int
            Index of the coefficient to compute inference for.

        Returns
        -------
        tuple
            (se, p_value, conf_int, t_stat, vcov, bootstrap_results)
        """
        bootstrap_results = wild_bootstrap_se(
            X, y, residuals, cluster_ids,
            coefficient_index=coefficient_index,
            n_bootstrap=self.n_bootstrap,
            weight_type=self.bootstrap_weights,
            alpha=self.alpha,
            seed=self.seed,
            return_distribution=False
        )
        self._bootstrap_results = bootstrap_results

        se = bootstrap_results.se
        p_value = bootstrap_results.p_value
        conf_int = (bootstrap_results.ci_lower, bootstrap_results.ci_upper)
        t_stat = bootstrap_results.t_stat_original

        # Also compute vcov for storage (using cluster-robust for consistency)
        vcov = compute_robust_se(X, residuals, cluster_ids)

        return se, p_value, conf_int, t_stat, vcov, bootstrap_results

    def _parse_formula(
        self, formula: str, data: pd.DataFrame
    ) -> Tuple[str, str, str, Optional[List[str]]]:
        """
        Parse R-style formula.

        Supports basic formulas like:
        - "outcome ~ treatment * time"
        - "outcome ~ treatment + time + treatment:time"
        - "outcome ~ treatment * time + covariate1 + covariate2"

        Parameters
        ----------
        formula : str
            R-style formula string.
        data : pd.DataFrame
            DataFrame to validate column names against.

        Returns
        -------
        tuple
            (outcome, treatment, time, covariates)
        """
        # Split into LHS and RHS
        if "~" not in formula:
            raise ValueError("Formula must contain '~' to separate outcome from predictors")

        lhs, rhs = formula.split("~")
        outcome = lhs.strip()

        # Parse RHS
        rhs = rhs.strip()

        # Check for interaction term
        if "*" in rhs:
            # Handle "treatment * time" syntax
            parts = rhs.split("*")
            if len(parts) != 2:
                raise ValueError("Currently only supports single interaction (treatment * time)")

            treatment = parts[0].strip()
            time = parts[1].strip()

            # Check for additional covariates after interaction
            if "+" in time:
                time_parts = time.split("+")
                time = time_parts[0].strip()
                covariates = [p.strip() for p in time_parts[1:]]
            else:
                covariates = None

        elif ":" in rhs:
            # Handle explicit interaction syntax
            terms = [t.strip() for t in rhs.split("+")]
            interaction_term = None
            main_effects = []
            covariates = []

            for term in terms:
                if ":" in term:
                    interaction_term = term
                else:
                    main_effects.append(term)

            if interaction_term is None:
                raise ValueError("Formula must contain an interaction term (treatment:time)")

            treatment, time = [t.strip() for t in interaction_term.split(":")]

            # Remaining terms after treatment and time are covariates
            for term in main_effects:
                if term != treatment and term != time:
                    covariates.append(term)

            covariates = covariates if covariates else None
        else:
            raise ValueError(
                "Formula must contain interaction term. "
                "Use 'outcome ~ treatment * time' or 'outcome ~ treatment + time + treatment:time'"
            )

        # Validate columns exist
        for col in [outcome, treatment, time]:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in data")

        if covariates:
            for cov in covariates:
                if cov not in data.columns:
                    raise ValueError(f"Covariate '{cov}' not found in data")

        return outcome, treatment, time, covariates

    def _validate_data(
        self,
        data: pd.DataFrame,
        outcome: str,
        treatment: str,
        time: str,
        covariates: Optional[List[str]] = None
    ) -> None:
        """Validate input data."""
        # Check DataFrame
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")

        # Check required columns exist
        required_cols = [outcome, treatment, time]
        if covariates:
            required_cols.extend(covariates)

        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in data: {missing_cols}")

        # Check for missing values
        for col in required_cols:
            if data[col].isna().any():
                raise ValueError(f"Column '{col}' contains missing values")

        # Check for sufficient variation
        if data[treatment].nunique() < 2:
            raise ValueError("Treatment variable must have both 0 and 1 values")
        if data[time].nunique() < 2:
            raise ValueError("Time variable must have both 0 and 1 values")

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Predict outcomes using fitted model.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with same structure as training data.

        Returns
        -------
        np.ndarray
            Predicted values.
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before calling predict()")

        # This is a placeholder - would need to store column names
        # for full implementation
        raise NotImplementedError(
            "predict() is not yet implemented. "
            "Use results_.fitted_values for training data predictions."
        )

    def get_params(self) -> Dict[str, Any]:
        """
        Get estimator parameters (sklearn-compatible).

        Returns
        -------
        Dict[str, Any]
            Estimator parameters.
        """
        return {
            "robust": self.robust,
            "cluster": self.cluster,
            "alpha": self.alpha,
            "inference": self.inference,
            "n_bootstrap": self.n_bootstrap,
            "bootstrap_weights": self.bootstrap_weights,
            "seed": self.seed,
        }

    def set_params(self, **params) -> "DifferenceInDifferences":
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


class MultiPeriodDiD(DifferenceInDifferences):
    """
    Multi-Period Difference-in-Differences estimator.

    Extends the standard DiD to handle multiple pre-treatment and
    post-treatment time periods, providing period-specific treatment
    effects as well as an aggregate average treatment effect.

    Parameters
    ----------
    robust : bool, default=True
        Whether to use heteroskedasticity-robust standard errors (HC1).
    cluster : str, optional
        Column name for cluster-robust standard errors.
    alpha : float, default=0.05
        Significance level for confidence intervals.

    Attributes
    ----------
    results_ : MultiPeriodDiDResults
        Estimation results after calling fit().
    is_fitted_ : bool
        Whether the model has been fitted.

    Examples
    --------
    Basic usage with multiple time periods:

    >>> import pandas as pd
    >>> from diff_diff import MultiPeriodDiD
    >>>
    >>> # Create sample panel data with 6 time periods
    >>> # Periods 0-2 are pre-treatment, periods 3-5 are post-treatment
    >>> data = create_panel_data()  # Your data
    >>>
    >>> # Fit the model
    >>> did = MultiPeriodDiD()
    >>> results = did.fit(
    ...     data,
    ...     outcome='sales',
    ...     treatment='treated',
    ...     time='period',
    ...     post_periods=[3, 4, 5]  # Specify which periods are post-treatment
    ... )
    >>>
    >>> # View period-specific effects
    >>> for period, effect in results.period_effects.items():
    ...     print(f"Period {period}: {effect.effect:.3f} (SE: {effect.se:.3f})")
    >>>
    >>> # View average treatment effect
    >>> print(f"Average ATT: {results.avg_att:.3f}")

    Notes
    -----
    The model estimates:

        Y_it = α + β*D_i + Σ_t γ_t*Period_t + Σ_t∈post δ_t*(D_i × Post_t) + ε_it

    Where:
    - D_i is the treatment indicator
    - Period_t are time period dummies
    - D_i × Post_t are treatment-by-post-period interactions
    - δ_t are the period-specific treatment effects

    The average ATT is computed as the mean of the δ_t coefficients.
    """

    def fit(  # type: ignore[override]
        self,
        data: pd.DataFrame,
        outcome: str,
        treatment: str,
        time: str,
        post_periods: Optional[List[Any]] = None,
        covariates: Optional[List[str]] = None,
        fixed_effects: Optional[List[str]] = None,
        absorb: Optional[List[str]] = None,
        reference_period: Any = None
    ) -> MultiPeriodDiDResults:
        """
        Fit the Multi-Period Difference-in-Differences model.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing the outcome, treatment, and time variables.
        outcome : str
            Name of the outcome variable column.
        treatment : str
            Name of the treatment group indicator column (0/1).
        time : str
            Name of the time period column (can have multiple values).
        post_periods : list
            List of time period values that are post-treatment.
            All other periods are treated as pre-treatment.
        covariates : list, optional
            List of covariate column names to include as linear controls.
        fixed_effects : list, optional
            List of categorical column names to include as fixed effects.
        absorb : list, optional
            List of categorical column names for high-dimensional fixed effects.
        reference_period : any, optional
            The reference (omitted) time period for the period dummies.
            Defaults to the first pre-treatment period.

        Returns
        -------
        MultiPeriodDiDResults
            Object containing period-specific and average treatment effects.

        Raises
        ------
        ValueError
            If required parameters are missing or data validation fails.
        """
        # Warn if wild bootstrap is requested but not supported
        if self.inference == "wild_bootstrap":
            import warnings
            warnings.warn(
                "Wild bootstrap inference is not yet supported for MultiPeriodDiD. "
                "Using analytical inference instead.",
                UserWarning
            )

        # Validate basic inputs
        if outcome is None or treatment is None or time is None:
            raise ValueError(
                "Must provide 'outcome', 'treatment', and 'time'"
            )

        # Validate columns exist
        self._validate_data(data, outcome, treatment, time, covariates)

        # Validate treatment is binary
        validate_binary(data[treatment].values, "treatment")

        # Get all unique time periods
        all_periods = sorted(data[time].unique())

        if len(all_periods) < 2:
            raise ValueError("Time variable must have at least 2 unique periods")

        # Determine pre and post periods
        if post_periods is None:
            # Default: last half of periods are post-treatment
            mid_point = len(all_periods) // 2
            post_periods = all_periods[mid_point:]
            pre_periods = all_periods[:mid_point]
        else:
            post_periods = list(post_periods)
            pre_periods = [p for p in all_periods if p not in post_periods]

        if len(post_periods) == 0:
            raise ValueError("Must have at least one post-treatment period")

        if len(pre_periods) == 0:
            raise ValueError("Must have at least one pre-treatment period")

        # Validate post_periods are in the data
        for p in post_periods:
            if p not in all_periods:
                raise ValueError(f"Post-period '{p}' not found in time column")

        # Determine reference period (omitted dummy)
        if reference_period is None:
            reference_period = pre_periods[0]
        elif reference_period not in all_periods:
            raise ValueError(f"Reference period '{reference_period}' not found in time column")

        # Validate fixed effects and absorb columns
        if fixed_effects:
            for fe in fixed_effects:
                if fe not in data.columns:
                    raise ValueError(f"Fixed effect column '{fe}' not found in data")
        if absorb:
            for ab in absorb:
                if ab not in data.columns:
                    raise ValueError(f"Absorb column '{ab}' not found in data")

        # Handle absorbed fixed effects (within-transformation)
        working_data = data.copy()
        n_absorbed_effects = 0

        if absorb:
            vars_to_demean = [outcome] + (covariates or [])
            for ab_var in absorb:
                n_absorbed_effects += working_data[ab_var].nunique() - 1
                for var in vars_to_demean:
                    group_means = working_data.groupby(ab_var)[var].transform("mean")
                    working_data[var] = working_data[var] - group_means

        # Extract outcome and treatment
        y = working_data[outcome].values.astype(float)
        d = working_data[treatment].values.astype(float)
        t = working_data[time].values

        # Build design matrix
        # Start with intercept and treatment main effect
        X = np.column_stack([np.ones(len(y)), d])
        var_names = ["const", treatment]

        # Add period dummies (excluding reference period)
        non_ref_periods = [p for p in all_periods if p != reference_period]
        period_dummy_indices = {}  # Map period -> column index in X

        for period in non_ref_periods:
            period_dummy = (t == period).astype(float)
            X = np.column_stack([X, period_dummy])
            var_names.append(f"period_{period}")
            period_dummy_indices[period] = X.shape[1] - 1

        # Add treatment × post-period interactions
        # These are our coefficients of interest
        interaction_indices = {}  # Map post-period -> column index in X

        for period in post_periods:
            interaction = d * (t == period).astype(float)
            X = np.column_stack([X, interaction])
            var_names.append(f"{treatment}:period_{period}")
            interaction_indices[period] = X.shape[1] - 1

        # Add covariates if provided
        if covariates:
            for cov in covariates:
                X = np.column_stack([X, working_data[cov].values.astype(float)])
                var_names.append(cov)

        # Add fixed effects as dummy variables
        if fixed_effects:
            for fe in fixed_effects:
                dummies = pd.get_dummies(working_data[fe], prefix=fe, drop_first=True)
                for col in dummies.columns:
                    X = np.column_stack([X, dummies[col].values.astype(float)])
                    var_names.append(col)

        # Fit OLS
        coefficients, residuals, fitted, r_squared = self._fit_ols(X, y)

        # Degrees of freedom
        df = len(y) - X.shape[1] - n_absorbed_effects

        # Compute standard errors
        # Note: Wild bootstrap for multi-period effects is complex (multiple coefficients)
        # For now, we use analytical inference even if inference="wild_bootstrap"
        if self.cluster is not None:
            cluster_ids = data[self.cluster].values
            vcov = compute_robust_se(X, residuals, cluster_ids)
        elif self.robust:
            vcov = compute_robust_se(X, residuals)
        else:
            n = len(y)
            k = X.shape[1]
            mse = np.sum(residuals ** 2) / (n - k)
            # Use solve() instead of inv() for numerical stability
            # solve(A, B) computes X where AX=B, so this yields (X'X)^{-1} * mse
            vcov = np.linalg.solve(X.T @ X, mse * np.eye(k))

        # Extract period-specific treatment effects
        period_effects = {}
        effect_values = []
        effect_indices = []

        for period in post_periods:
            idx = interaction_indices[period]
            effect = coefficients[idx]
            se = np.sqrt(vcov[idx, idx])
            t_stat = effect / se
            p_value = compute_p_value(t_stat, df=df)
            conf_int = compute_confidence_interval(effect, se, self.alpha, df=df)

            period_effects[period] = PeriodEffect(
                period=period,
                effect=effect,
                se=se,
                t_stat=t_stat,
                p_value=p_value,
                conf_int=conf_int
            )
            effect_values.append(effect)
            effect_indices.append(idx)

        # Compute average treatment effect
        # Average ATT = mean of period-specific effects
        avg_att = np.mean(effect_values)

        # Standard error of average: need to account for covariance
        # Var(avg) = (1/n^2) * sum of all elements in the sub-covariance matrix
        n_post = len(post_periods)
        sub_vcov = vcov[np.ix_(effect_indices, effect_indices)]
        avg_var = np.sum(sub_vcov) / (n_post ** 2)
        avg_se = np.sqrt(avg_var)

        avg_t_stat = avg_att / avg_se if avg_se > 0 else 0.0
        avg_p_value = compute_p_value(avg_t_stat, df=df)
        avg_conf_int = compute_confidence_interval(avg_att, avg_se, self.alpha, df=df)

        # Count observations
        n_treated = int(np.sum(d))
        n_control = int(np.sum(1 - d))

        # Create coefficient dictionary
        coef_dict = {name: coef for name, coef in zip(var_names, coefficients)}

        # Store results
        self.results_ = MultiPeriodDiDResults(
            period_effects=period_effects,
            avg_att=avg_att,
            avg_se=avg_se,
            avg_t_stat=avg_t_stat,
            avg_p_value=avg_p_value,
            avg_conf_int=avg_conf_int,
            n_obs=len(y),
            n_treated=n_treated,
            n_control=n_control,
            pre_periods=pre_periods,
            post_periods=post_periods,
            alpha=self.alpha,
            coefficients=coef_dict,
            vcov=vcov,
            residuals=residuals,
            fitted_values=fitted,
            r_squared=r_squared,
        )

        self._coefficients = coefficients
        self._vcov = vcov
        self.is_fitted_ = True

        return self.results_

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


# Re-export estimators from submodules for backward compatibility
# These can also be imported directly from their respective modules:
# - from diff_diff.twfe import TwoWayFixedEffects
# - from diff_diff.synthetic_did import SyntheticDiD
from diff_diff.synthetic_did import SyntheticDiD  # noqa: E402
from diff_diff.twfe import TwoWayFixedEffects  # noqa: E402

__all__ = [
    "DifferenceInDifferences",
    "MultiPeriodDiD",
    "TwoWayFixedEffects",
    "SyntheticDiD",
]
