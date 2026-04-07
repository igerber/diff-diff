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
from diff_diff.linalg import LinearRegression
from diff_diff.results import DiDResults
from diff_diff.utils import (
    within_transform as _within_transform_util,
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
        covariates: Optional[List[str]] = None,
        survey_design: object = None,
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
        survey_design : SurveyDesign, optional
            Survey design specification for design-based inference. When provided,
            uses Taylor Series Linearization for variance estimation and
            applies sampling weights to the regression.

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

        # Warn if time has more than 2 unique values (not a binary post indicator)
        n_unique_time = data[time].nunique()
        if n_unique_time > 2:
            warnings.warn(
                f"The '{time}' column has {n_unique_time} unique values. "
                f"TwoWayFixedEffects expects a binary (0/1) post indicator. "
                f"Multi-period time values produce 'treated * period_number' instead of "
                f"'treated * post_indicator', which may not estimate the standard DiD ATT. "
                f"Consider creating a binary post column: "
                f"df['post'] = (df['{time}'] >= cutoff).astype(int)",
                UserWarning,
                stacklevel=2,
            )
        elif n_unique_time == 2:
            unique_vals = set(data[time].unique())
            if unique_vals != {0, 1} and unique_vals != {False, True}:
                warnings.warn(
                    f"The '{time}' column has values {sorted(unique_vals)} instead of {{0, 1}}. "
                    f"The ATT estimate is mathematically correct (within-transformation "
                    f"absorbs the scaling), but 0/1 encoding is recommended for clarity. "
                    f"Consider: df['{time}'] = (df['{time}'] == {max(unique_vals)}).astype(int)",
                    UserWarning,
                    stacklevel=2,
                )

        # Resolve survey design if provided
        from diff_diff.survey import _resolve_effective_cluster, _resolve_survey_for_fit

        resolved_survey, survey_weights, survey_weight_type, survey_metadata = (
            _resolve_survey_for_fit(survey_design, data, self.inference)
        )
        # Reject replicate-weight designs — TWFE within-transformation must
        # be recomputed per replicate (same reason as SunAbraham rejection)
        if resolved_survey is not None and resolved_survey.uses_replicate_variance:
            raise NotImplementedError(
                "TwoWayFixedEffects does not yet support replicate-weight "
                "survey designs. The weighted within-transformation must be "
                "recomputed for each replicate. Use CallawaySantAnna or "
                "TripleDifference for replicate-weight inference, or use a "
                "TSL-based survey design (strata/psu/fpc)."
            )

        # Use unit-level clustering if not specified (use local variable to avoid mutation)
        cluster_var = self.cluster if self.cluster is not None else unit

        # Create treatment × post interaction from raw data before demeaning.
        # This must be within-transformed alongside the outcome and covariates
        # so that the regression uses demeaned regressors (FWL theorem).
        data = data.copy()
        data["_treatment_post"] = data[treatment] * data[time]

        # Demean outcome, covariates, AND interaction in a single pass
        all_vars = [outcome] + (covariates or []) + ["_treatment_post"]
        data_demeaned = _within_transform_util(
            data,
            all_vars,
            unit,
            time,
            suffix="_demeaned",
            weights=survey_weights,
        )

        # Extract variables for regression
        y = data_demeaned[f"{outcome}_demeaned"].values
        X_list = [data_demeaned["_treatment_post_demeaned"].values]

        if covariates:
            for cov in covariates:
                X_list.append(data_demeaned[f"{cov}_demeaned"].values)

        X = np.column_stack([np.ones(len(y))] + X_list)

        # ATT is the coefficient on treatment_post (index 1)
        att_idx = 1

        # Degrees of freedom adjustment for fixed effects
        n_units = data[unit].nunique()
        n_times = data[time].nunique()
        df_adjustment = n_units + n_times - 2

        # Always use LinearRegression for initial fit (unified code path)
        # For wild bootstrap, we don't need cluster SEs from the initial fit
        cluster_ids = data[cluster_var].values

        # When survey PSU is present, it overrides cluster for variance estimation
        effective_cluster_ids = _resolve_effective_cluster(
            resolved_survey, cluster_ids, self.cluster
        )

        # For survey variance: only inject user-explicit cluster as PSU.
        # TWFE's default unit clustering should not override the documented
        # no-PSU survey path (implicit per-observation PSUs).
        if resolved_survey is not None and self.cluster is None:
            survey_cluster_ids = None
        else:
            survey_cluster_ids = effective_cluster_ids

        # Inject cluster as effective PSU for survey variance estimation
        if resolved_survey is not None and survey_cluster_ids is not None:
            from diff_diff.survey import _inject_cluster_as_psu, compute_survey_metadata

            resolved_survey = _inject_cluster_as_psu(resolved_survey, survey_cluster_ids)
            if resolved_survey.psu is not None and survey_metadata is not None:
                raw_w = (
                    data[survey_design.weights].values.astype(np.float64)
                    if survey_design.weights
                    else np.ones(len(data), dtype=np.float64)
                )
                survey_metadata = compute_survey_metadata(resolved_survey, raw_w)

        # Pass rank_deficient_action to LinearRegression
        # If "error", let LinearRegression raise immediately
        # If "warn" or "silent", suppress generic warning and use TWFE's context-specific
        # error/warning messages (more informative for panel data)
        if self.rank_deficient_action == "error":
            reg = LinearRegression(
                include_intercept=False,
                robust=True,
                cluster_ids=survey_cluster_ids if self.inference != "wild_bootstrap" else None,
                alpha=self.alpha,
                rank_deficient_action="error",
                weights=survey_weights,
                weight_type=survey_weight_type,
                survey_design=resolved_survey,
            ).fit(X, y, df_adjustment=df_adjustment)
        else:
            # Suppress generic warning, TWFE provides context-specific messages below
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Rank-deficient design matrix")
                reg = LinearRegression(
                    include_intercept=False,
                    robust=True,
                    cluster_ids=(
                        survey_cluster_ids if self.inference != "wild_bootstrap" else None
                    ),
                    alpha=self.alpha,
                    rank_deficient_action="silent",
                    weights=survey_weights,
                    weight_type=survey_weight_type,
                    survey_design=resolved_survey,
                ).fit(X, y, df_adjustment=df_adjustment)

        coefficients = reg.coefficients_
        residuals = reg.residuals_
        fitted = reg.fitted_values_
        r_squared = reg.r_squared()
        assert coefficients is not None
        att = coefficients[att_idx]

        # Check for unidentified coefficients (collinearity)
        # Build column names for informative error messages
        column_names = ["intercept", "treatment×post"]
        if covariates:
            column_names.extend(covariates)

        nan_mask = np.isnan(coefficients)
        if np.any(nan_mask):
            dropped_indices = np.where(nan_mask)[0]
            dropped_names = [
                column_names[i] if i < len(column_names) else f"column {i}" for i in dropped_indices
            ]

            # Determine the source of collinearity for better error message
            if att_idx in dropped_indices:
                # Treatment coefficient is unidentified
                raise ValueError(
                    f"Treatment effect cannot be identified due to collinearity. "
                    f"Dropped columns: {', '.join(dropped_names)}. "
                    "This can happen when: (1) treatment is perfectly collinear with "
                    "unit/time fixed effects, (2) all treated units are treated in all "
                    "periods, or (3) a covariate is collinear with the treatment indicator. "
                    "Check your data structure and model specification."
                )
            else:
                # Only covariates are dropped - this is a warning, not an error
                # The ATT can still be estimated
                # Respect rank_deficient_action setting for warning
                if self.rank_deficient_action == "warn":
                    warnings.warn(
                        f"Some covariates are collinear and were dropped: "
                        f"{', '.join(dropped_names)}. The treatment effect is still identified.",
                        UserWarning,
                        stacklevel=2,
                    )

        # Get inference - either from bootstrap or analytical
        if self.inference == "wild_bootstrap":
            # Override with wild cluster bootstrap inference
            se, p_value, conf_int, t_stat, vcov, _ = self._run_wild_bootstrap_inference(
                X, y, residuals, cluster_ids, att_idx
            )
        else:
            # Use analytical inference from LinearRegression
            vcov = reg.vcov_
            inference = reg.get_inference(att_idx)
            se = inference.se
            t_stat = inference.t_stat
            p_value = inference.p_value
            conf_int = inference.conf_int

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
            survey_metadata=survey_metadata,
        )

        self.is_fitted_ = True
        return self.results_

    def _within_transform(
        self,
        data: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        covariates: Optional[List[str]] = None,
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
        variables = [outcome] + (covariates or [])
        return _within_transform_util(data, variables, unit, time, suffix="_demeaned")

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

        Note: This check requires ``time`` to have actual period values (not
        binary 0/1). With binary time, all treated units appear to start at
        time=1, so staggering is undetectable.
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
