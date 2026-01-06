"""
Two-Way Fixed Effects estimator for panel Difference-in-Differences.
"""

import warnings
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from diff_diff.bacon import BaconDecompositionResults

from diff_diff.estimators import DifferenceInDifferences
from diff_diff.results import DiDResults
from diff_diff.utils import (
    compute_confidence_interval,
    compute_p_value,
    compute_robust_se,
)


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
        If None, automatically clusters at the unit level (the `unit`
        parameter passed to `fit()`). This differs from
        DifferenceInDifferences where cluster=None means no clustering.
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

    def fit(  # type: ignore[override]
        self,
        data: pd.DataFrame,
        outcome: str,
        treatment: str,
        time: str,
        unit: str,
        covariates: Optional[List[str]] = None
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

        # Check for staggered treatment timing and warn if detected
        self._check_staggered_treatment(data, treatment, time, unit)

        # Use unit-level clustering if not specified (use local variable to avoid mutation)
        cluster_var = self.cluster if self.cluster is not None else unit

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
        att_idx = 1
        att = coefficients[att_idx]

        # Degrees of freedom adjustment for fixed effects
        n_units = data[unit].nunique()
        n_times = data[time].nunique()
        df = len(y) - X.shape[1] - n_units - n_times + 2

        # Compute standard errors and inference
        cluster_ids = data[cluster_var].values
        if self.inference == "wild_bootstrap":
            # Wild cluster bootstrap for few-cluster inference
            se, p_value, conf_int, t_stat, vcov, _ = self._run_wild_bootstrap_inference(
                X, y, residuals, cluster_ids, att_idx
            )
        else:
            # Standard cluster-robust SE
            vcov = compute_robust_se(X, residuals, cluster_ids)
            se = np.sqrt(vcov[att_idx, att_idx])
            t_stat = att / se
            p_value = compute_p_value(t_stat, df=df)
            conf_int = compute_confidence_interval(att, se, self.alpha, df=df)

        # Count observations
        treated_units = data[data[treatment] == 1][unit].unique()
        n_treated = len(treated_units)
        n_control = n_units - n_treated

        # Determine inference method and bootstrap info
        inference_method = "analytical"
        n_bootstrap_used = None
        n_clusters_used = None
        if self._bootstrap_results is not None:
            inference_method = "wild_bootstrap"
            n_bootstrap_used = self._bootstrap_results.n_bootstrap
            n_clusters_used = self._bootstrap_results.n_clusters

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
            coefficients={"ATT": float(att)},
            vcov=vcov,
            residuals=residuals,
            fitted_values=fitted,
            r_squared=r_squared,
            inference_method=inference_method,
            n_bootstrap=n_bootstrap_used,
            n_clusters=n_clusters_used,
        )

        self.is_fitted_ = True
        return self.results_

    def _within_transform(
        self,
        data: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        covariates: Optional[List[str]] = None
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

    def _check_staggered_treatment(
        self,
        data: pd.DataFrame,
        treatment: str,
        time: str,
        unit: str,
    ) -> None:
        """
        Check for staggered treatment timing and warn if detected.

        Identifies if different units start treatment at different times,
        which can bias TWFE estimates when treatment effects are heterogeneous.
        """
        # Find first treatment time for each unit
        treated_obs = data[data[treatment] == 1]
        if len(treated_obs) == 0:
            return  # No treated observations

        # Get first treatment time per unit
        first_treat_times = treated_obs.groupby(unit)[time].min()
        unique_treat_times = first_treat_times.unique()

        if len(unique_treat_times) > 1:
            n_groups = len(unique_treat_times)
            warnings.warn(
                f"Staggered treatment timing detected: {n_groups} treatment cohorts "
                f"start treatment at different times. TWFE can be biased when treatment "
                f"effects are heterogeneous across time. Consider using:\n"
                f"  - CallawaySantAnna estimator for robust estimates\n"
                f"  - TwoWayFixedEffects.decompose() to diagnose the decomposition\n"
                f"  - bacon_decompose() to see weight on 'forbidden' comparisons",
                UserWarning,
                stacklevel=3,
            )

    def decompose(
        self,
        data: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        first_treat: str,
        weights: str = "approximate",
    ) -> "BaconDecompositionResults":
        """
        Perform Goodman-Bacon decomposition of TWFE estimate.

        Decomposes the TWFE estimate into a weighted average of all possible
        2x2 DiD comparisons, revealing which comparisons drive the estimate
        and whether problematic "forbidden comparisons" are involved.

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
            Name of column indicating when each unit was first treated.
            Use 0 (or np.inf) for never-treated units.
        weights : str, default="approximate"
            Weight calculation method:
            - "approximate": Fast simplified formula (default). Good for
              diagnostic purposes where relative weights are sufficient.
            - "exact": Variance-based weights from Goodman-Bacon (2021)
              Theorem 1. Use for publication-quality decompositions.

        Returns
        -------
        BaconDecompositionResults
            Decomposition results showing:
            - TWFE estimate and its weighted-average breakdown
            - List of all 2x2 comparisons with estimates and weights
            - Total weight by comparison type (clean vs forbidden)

        Examples
        --------
        >>> twfe = TwoWayFixedEffects()
        >>> decomp = twfe.decompose(
        ...     data, outcome='y', unit='id', time='t', first_treat='treat_year'
        ... )
        >>> decomp.print_summary()
        >>> # Check weight on forbidden comparisons
        >>> if decomp.total_weight_later_vs_earlier > 0.2:
        ...     print("Warning: significant forbidden comparison weight")

        Notes
        -----
        This decomposition is essential for understanding potential TWFE bias
        in staggered adoption designs. The three comparison types are:

        1. **Treated vs Never-treated**: Clean comparisons using never-treated
           units as controls. These are always valid.

        2. **Earlier vs Later treated**: Uses later-treated units as controls
           before they receive treatment. These are valid.

        3. **Later vs Earlier treated**: Uses already-treated units as controls.
           These "forbidden comparisons" can introduce bias when treatment
           effects are dynamic (changing over time since treatment).

        See Also
        --------
        bacon_decompose : Standalone decomposition function
        BaconDecomposition : Class-based decomposition interface
        CallawaySantAnna : Robust estimator that avoids forbidden comparisons
        """
        from diff_diff.bacon import BaconDecomposition

        decomp = BaconDecomposition(weights=weights)
        return decomp.fit(data, outcome, unit, time, first_treat)
