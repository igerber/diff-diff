"""
Difference-in-Differences estimators with sklearn-like API.
"""

from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

from diff_diff.results import DiDResults
from diff_diff.utils import (
    validate_binary,
    compute_robust_se,
    compute_confidence_interval,
    compute_p_value,
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
        cluster: str = None,
        alpha: float = 0.05
    ):
        self.robust = robust
        self.cluster = cluster
        self.alpha = alpha

        self.is_fitted_ = False
        self.results_ = None
        self._coefficients = None
        self._vcov = None

    def fit(
        self,
        data: pd.DataFrame,
        outcome: str = None,
        treatment: str = None,
        time: str = None,
        formula: str = None,
        covariates: list = None,
        fixed_effects: list = None,
        absorb: list = None
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
            vars_to_demean = [outcome] + (covariates or [])
            for ab_var in absorb:
                n_absorbed_effects += working_data[ab_var].nunique() - 1
                for var in vars_to_demean:
                    group_means = working_data.groupby(ab_var)[var].transform("mean")
                    working_data[var] = working_data[var] - group_means
                absorbed_vars.append(ab_var)

        # Extract variables
        y = working_data[outcome].values.astype(float)
        d = working_data[treatment].values.astype(float)
        t = working_data[time].values.astype(float)

        # Validate binary variables
        validate_binary(d, "treatment")
        validate_binary(t, "time")

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
                dummies = pd.get_dummies(data[fe], prefix=fe, drop_first=True)
                for col in dummies.columns:
                    X = np.column_stack([X, dummies[col].values.astype(float)])
                    var_names.append(col)

        # Fit OLS
        coefficients, residuals, fitted, r_squared = self._fit_ols(X, y)

        # Compute standard errors
        if self.cluster is not None:
            cluster_ids = data[self.cluster].values
            vcov = compute_robust_se(X, residuals, cluster_ids)
        elif self.robust:
            vcov = compute_robust_se(X, residuals)
        else:
            # Classical OLS standard errors
            n = len(y)
            k = X.shape[1]
            mse = np.sum(residuals ** 2) / (n - k)
            vcov = mse * np.linalg.inv(X.T @ X)

        # Extract ATT (coefficient on interaction term)
        att_idx = 3  # Index of interaction term
        att = coefficients[att_idx]
        se = np.sqrt(vcov[att_idx, att_idx])

        # Compute test statistics (adjust df for absorbed fixed effects)
        df = len(y) - X.shape[1] - n_absorbed_effects
        t_stat = att / se
        p_value = compute_p_value(t_stat, df=df)
        conf_int = compute_confidence_interval(att, se, self.alpha, df=df)

        # Count observations
        n_treated = int(np.sum(d))
        n_control = int(np.sum(1 - d))

        # Create coefficient dictionary
        coef_dict = {name: coef for name, coef in zip(var_names, coefficients)}

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
        )

        self._coefficients = coefficients
        self._vcov = vcov
        self.is_fitted_ = True

        return self.results_

    def _fit_ols(self, X: np.ndarray, y: np.ndarray) -> tuple:
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
        """
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

    def _parse_formula(
        self, formula: str, data: pd.DataFrame
    ) -> tuple:
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
        covariates: list = None
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

    def get_params(self) -> dict:
        """
        Get estimator parameters (sklearn-compatible).

        Returns
        -------
        dict
            Estimator parameters.
        """
        return {
            "robust": self.robust,
            "cluster": self.cluster,
            "alpha": self.alpha,
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
        return self.results_.summary()

    def print_summary(self) -> None:
        """Print summary to stdout."""
        print(self.summary())


class TwoWayFixedEffects(DifferenceInDifferences):
    """
    Two-Way Fixed Effects (TWFE) estimator for panel DiD.

    Extends DifferenceInDifferences to handle panel data with unit
    and time fixed effects.

    Parameters
    ----------
    robust : bool, default=True
        Whether to use heteroskedasticity-robust standard errors.
    cluster : str, optional
        Column name for cluster-robust standard errors.
        Defaults to clustering at the unit level.
    alpha : float, default=0.05
        Significance level for confidence intervals.

    Notes
    -----
    This estimator uses the regression:

        Y_it = α_i + γ_t + β*(D_i × Post_t) + X_it'δ + ε_it

    where α_i are unit fixed effects and γ_t are time fixed effects.

    Warning: TWFE can be biased with staggered treatment timing
    and heterogeneous treatment effects. Consider using
    more robust estimators (e.g., Callaway-Sant'Anna) for
    staggered designs.
    """

    def fit(
        self,
        data: pd.DataFrame,
        outcome: str,
        treatment: str,
        time: str,
        unit: str,
        covariates: list = None
    ) -> DiDResults:
        """
        Fit Two-Way Fixed Effects model.

        Parameters
        ----------
        data : pd.DataFrame
            Panel data.
        outcome : str
            Name of outcome variable column.
        treatment : str
            Name of treatment indicator column.
        time : str
            Name of time period column.
        unit : str
            Name of unit identifier column.
        covariates : list, optional
            List of covariate column names.

        Returns
        -------
        DiDResults
            Estimation results.
        """
        # Validate unit column exists
        if unit not in data.columns:
            raise ValueError(f"Unit column '{unit}' not found in data")

        # Set cluster to unit level if not specified
        if self.cluster is None:
            self.cluster = unit

        # Demean data (within transformation for fixed effects)
        data_demeaned = self._within_transform(data, outcome, unit, time, covariates)

        # Create treatment × post interaction
        # For staggered designs, we'd need to identify treatment timing per unit
        # For now, assume standard 2-period design
        data_demeaned["_treatment_post"] = (
            data_demeaned[treatment] * data_demeaned[time]
        )

        # Extract variables for regression
        y = data_demeaned[f"{outcome}_demeaned"].values
        X_list = [data_demeaned["_treatment_post"].values]

        if covariates:
            for cov in covariates:
                X_list.append(data_demeaned[f"{cov}_demeaned"].values)

        X = np.column_stack([np.ones(len(y))] + X_list)

        # Fit OLS on demeaned data
        coefficients, residuals, fitted, r_squared = self._fit_ols(X, y)

        # ATT is the coefficient on treatment_post (index 1)
        att = coefficients[1]

        # Compute cluster-robust standard errors
        cluster_ids = data[self.cluster].values
        vcov = compute_robust_se(X, residuals, cluster_ids)
        se = np.sqrt(vcov[1, 1])

        # Degrees of freedom adjustment for fixed effects
        n_units = data[unit].nunique()
        n_times = data[time].nunique()
        df = len(y) - X.shape[1] - n_units - n_times + 2

        t_stat = att / se
        p_value = compute_p_value(t_stat, df=df)
        conf_int = compute_confidence_interval(att, se, self.alpha, df=df)

        # Count observations
        treated_units = data[data[treatment] == 1][unit].unique()
        n_treated = len(treated_units)
        n_control = n_units - n_treated

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
            coefficients={"ATT": att},
            vcov=vcov,
            residuals=residuals,
            fitted_values=fitted,
            r_squared=r_squared,
        )

        self.is_fitted_ = True
        return self.results_

    def _within_transform(
        self,
        data: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        covariates: list = None
    ) -> pd.DataFrame:
        """
        Apply within transformation to remove unit and time fixed effects.

        This implements the standard two-way within transformation:
        y_it - y_i. - y_.t + y_..

        Parameters
        ----------
        data : pd.DataFrame
            Panel data.
        outcome : str
            Outcome variable name.
        unit : str
            Unit identifier column.
        time : str
            Time period column.
        covariates : list, optional
            Covariate column names.

        Returns
        -------
        pd.DataFrame
            Data with demeaned variables.
        """
        data = data.copy()
        variables = [outcome] + (covariates or [])

        for var in variables:
            # Unit means
            unit_means = data.groupby(unit)[var].transform("mean")
            # Time means
            time_means = data.groupby(time)[var].transform("mean")
            # Grand mean
            grand_mean = data[var].mean()

            # Within transformation
            data[f"{var}_demeaned"] = data[var] - unit_means - time_means + grand_mean

        return data
