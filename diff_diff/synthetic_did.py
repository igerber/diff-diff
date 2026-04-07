"""
Synthetic Difference-in-Differences estimator.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError

from diff_diff.estimators import DifferenceInDifferences
from diff_diff.linalg import solve_ols
from diff_diff.results import SyntheticDiDResults
from diff_diff.utils import (
    _compute_regularization,
    _sum_normalize,
    compute_sdid_estimator,
    compute_sdid_unit_weights,
    compute_time_weights,
    safe_inference,
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
    zeta_omega : float, optional
        Regularization for unit weights. If None (default), auto-computed
        from data as ``(N1 * T1)^(1/4) * noise_level`` matching R's synthdid.
    zeta_lambda : float, optional
        Regularization for time weights. If None (default), auto-computed
        from data as ``1e-6 * noise_level`` matching R's synthdid.
    alpha : float, default=0.05
        Significance level for confidence intervals.
    variance_method : str, default="placebo"
        Method for variance estimation:
        - "placebo": Placebo-based variance matching R's synthdid::vcov(method="placebo").
          Implements Algorithm 4 from Arkhangelsky et al. (2021). This is R's default.
        - "bootstrap": Bootstrap at unit level with fixed weights matching R's
          synthdid::vcov(method="bootstrap").
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
        zeta_omega: Optional[float] = None,
        zeta_lambda: Optional[float] = None,
        alpha: float = 0.05,
        variance_method: str = "placebo",
        n_bootstrap: int = 200,
        seed: Optional[int] = None,
        # Deprecated — accepted for backward compat, ignored with warning
        lambda_reg: Optional[float] = None,
        zeta: Optional[float] = None,
    ):
        if lambda_reg is not None:
            warnings.warn(
                "lambda_reg is deprecated and ignored. Regularization is now "
                "auto-computed from data. Use zeta_omega to override unit weight "
                "regularization.",
                DeprecationWarning,
                stacklevel=2,
            )
        if zeta is not None:
            warnings.warn(
                "zeta is deprecated and ignored. Use zeta_lambda to override "
                "time weight regularization.",
                DeprecationWarning,
                stacklevel=2,
            )

        super().__init__(robust=True, cluster=None, alpha=alpha)
        self.zeta_omega = zeta_omega
        self.zeta_lambda = zeta_lambda
        self.variance_method = variance_method
        self.n_bootstrap = n_bootstrap
        self.seed = seed

        # Validate n_bootstrap
        if n_bootstrap < 2:
            raise ValueError(
                f"n_bootstrap must be >= 2 (got {n_bootstrap}). At least 2 "
                f"iterations are needed to estimate standard errors."
            )

        # Validate variance_method
        valid_methods = ("bootstrap", "placebo")
        if variance_method not in valid_methods:
            raise ValueError(
                f"variance_method must be one of {valid_methods}, " f"got '{variance_method}'"
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
        covariates: Optional[List[str]] = None,
        survey_design=None,
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
        survey_design : SurveyDesign, optional
            Survey design specification. Only pweight weight_type is supported.
            Strata/PSU/FPC are supported via Rao-Wu rescaled bootstrap when
            variance_method='bootstrap'. Placebo variance does not support
            strata/PSU/FPC; use variance_method='bootstrap' for full designs.

        Returns
        -------
        SyntheticDiDResults
            Object containing the ATT estimate, standard error,
            unit weights, and time weights.

        Raises
        ------
        ValueError
            If required parameters are missing, data validation fails,
            or a non-pweight survey design is provided.
        """
        # Validate inputs
        if outcome is None or treatment is None or unit is None or time is None:
            raise ValueError("Must provide 'outcome', 'treatment', 'unit', and 'time'")

        # Check columns exist
        required_cols = [outcome, treatment, unit, time]
        if covariates:
            required_cols.extend(covariates)

        missing = [c for c in required_cols if c not in data.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # Resolve survey design
        from diff_diff.survey import (
            _extract_unit_survey_weights,
            _resolve_survey_for_fit,
            _validate_unit_constant_survey,
        )

        resolved_survey, survey_weights, survey_weight_type, survey_metadata = (
            _resolve_survey_for_fit(survey_design, data, "analytical")
        )
        # Reject replicate-weight designs — SyntheticDiD uses bootstrap variance
        if resolved_survey is not None and resolved_survey.uses_replicate_variance:
            raise NotImplementedError(
                "SyntheticDiD does not yet support replicate-weight survey "
                "designs. Use a TSL-based survey design (strata/psu/fpc)."
            )
        # Validate pweight only (strata/PSU/FPC are allowed for Rao-Wu bootstrap)
        if resolved_survey is not None and resolved_survey.weight_type != "pweight":
            raise ValueError(
                "SyntheticDiD survey support requires weight_type='pweight'. "
                f"Got '{resolved_survey.weight_type}'."
            )

        # Reject placebo + full survey design (strata/PSU/FPC are silently ignored)
        if (
            resolved_survey is not None
            and (
                resolved_survey.strata is not None
                or resolved_survey.psu is not None
                or resolved_survey.fpc is not None
            )
            and self.variance_method == "placebo"
        ):
            raise NotImplementedError(
                "SyntheticDiD with variance_method='placebo' does not support strata/PSU/FPC. "
                "Use variance_method='bootstrap' for full survey design support."
            )

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

        # Validate treatment is constant within unit (SDID requires block treatment)
        treatment_nunique = data.groupby(unit)[treatment].nunique()
        varying_units = treatment_nunique[treatment_nunique > 1]
        if len(varying_units) > 0:
            example_unit = varying_units.index[0]
            example_vals = sorted(data.loc[data[unit] == example_unit, treatment].unique())
            raise ValueError(
                f"Treatment indicator varies within {len(varying_units)} unit(s) "
                f"(e.g., unit '{example_unit}' has values {example_vals}). "
                f"SyntheticDiD requires 'block' treatment where treatment is "
                f"constant within each unit across all time periods. "
                f"For staggered adoption designs, use CallawaySantAnna or "
                f"ImputationDiD instead."
            )

        treated_units = unit_treatment[unit_treatment == 1].index.tolist()
        control_units = unit_treatment[unit_treatment == 0].index.tolist()

        if len(treated_units) == 0:
            raise ValueError("No treated units found")
        if len(control_units) == 0:
            raise ValueError("No control units found")

        # Validate balanced panel (SDID requires all units observed in all periods)
        periods_per_unit = data.groupby(unit)[time].nunique()
        expected_n_periods = len(all_periods)
        unbalanced_units = periods_per_unit[periods_per_unit != expected_n_periods]
        if len(unbalanced_units) > 0:
            example_unit = unbalanced_units.index[0]
            actual_count = unbalanced_units.iloc[0]
            raise ValueError(
                f"Panel is not balanced: {len(unbalanced_units)} unit(s) do not "
                f"have observations in all {expected_n_periods} periods "
                f"(e.g., unit '{example_unit}' has {actual_count} periods). "
                f"SyntheticDiD requires a balanced panel. Use "
                f"diff_diff.prep.balance_panel() to balance the panel first."
            )

        # Validate and extract survey weights
        # Build unit-level ResolvedSurveyDesign for Rao-Wu bootstrap when
        # strata/PSU/FPC are present (survey columns are unit-constant).
        _unit_resolved_survey = None
        if resolved_survey is not None:
            _validate_unit_constant_survey(data, unit, survey_design)
            w_treated = _extract_unit_survey_weights(data, unit, survey_design, treated_units)
            w_control = _extract_unit_survey_weights(data, unit, survey_design, control_units)

            # Build unit-level resolved survey for Rao-Wu bootstrap
            _has_design = (
                resolved_survey.strata is not None
                or resolved_survey.psu is not None
                or resolved_survey.fpc is not None
            )
            if _has_design:
                _unit_resolved_survey = self._build_unit_resolved_survey(
                    data,
                    unit,
                    survey_design,
                    control_units,
                    treated_units,
                )
        else:
            w_treated = None
            w_control = None

        # Residualize covariates if provided
        working_data = data.copy()
        if covariates:
            working_data = self._residualize_covariates(
                working_data,
                outcome,
                covariates,
                unit,
                time,
                survey_weights=survey_weights,
                survey_weight_type=survey_weight_type,
            )

        # Create outcome matrices
        # Shape: (n_periods, n_units)
        Y_pre_control, Y_post_control, Y_pre_treated, Y_post_treated = (
            self._create_outcome_matrices(
                working_data,
                outcome,
                unit,
                time,
                pre_periods,
                post_periods,
                treated_units,
                control_units,
            )
        )

        # Compute auto-regularization (or use user overrides)
        auto_zeta_omega, auto_zeta_lambda = _compute_regularization(
            Y_pre_control, len(treated_units), len(post_periods)
        )
        zeta_omega = self.zeta_omega if self.zeta_omega is not None else auto_zeta_omega
        zeta_lambda = self.zeta_lambda if self.zeta_lambda is not None else auto_zeta_lambda

        # Store noise level for diagnostics
        from diff_diff.utils import _compute_noise_level

        noise_level = _compute_noise_level(Y_pre_control)

        # Data-dependent convergence threshold (matches R's 1e-5 * noise.level).
        # Floor of 1e-5 when noise_level == 0: R would use 0.0, causing FW to
        # run all max_iter iterations.  The result is equivalent (zero-noise
        # data has no variation to optimize), but the floor enables early stop.
        min_decrease = 1e-5 * noise_level if noise_level > 0 else 1e-5

        # Compute unit weights (Frank-Wolfe with sparsification)
        # Survey weights enter via the treated mean target
        if w_treated is not None:
            Y_pre_treated_mean = np.average(Y_pre_treated, axis=1, weights=w_treated)
        else:
            Y_pre_treated_mean = np.mean(Y_pre_treated, axis=1)

        unit_weights = compute_sdid_unit_weights(
            Y_pre_control,
            Y_pre_treated_mean,
            zeta_omega=zeta_omega,
            min_decrease=min_decrease,
        )

        # Compute time weights (Frank-Wolfe on collapsed form)
        time_weights = compute_time_weights(
            Y_pre_control,
            Y_post_control,
            zeta_lambda=zeta_lambda,
            min_decrease=min_decrease,
        )

        # Compose ω with control survey weights (WLS regression interpretation).
        # Frank-Wolfe finds best trajectory match; survey weights reweight by
        # population importance post-optimization.
        if w_control is not None:
            omega_eff = unit_weights * w_control
            omega_eff = omega_eff / omega_eff.sum()
        else:
            omega_eff = unit_weights

        # Compute SDID estimate
        if w_treated is not None:
            Y_post_treated_mean = np.average(Y_post_treated, axis=1, weights=w_treated)
        else:
            Y_post_treated_mean = np.mean(Y_post_treated, axis=1)

        att = compute_sdid_estimator(
            Y_pre_control,
            Y_post_control,
            Y_pre_treated_mean,
            Y_post_treated_mean,
            omega_eff,
            time_weights,
        )

        # Compute pre-treatment fit (RMSE) using composed weights
        synthetic_pre = Y_pre_control @ omega_eff
        pre_fit_rmse = np.sqrt(np.mean((Y_pre_treated_mean - synthetic_pre) ** 2))

        # Warn if pre-treatment fit is poor (Registry requirement).
        # Threshold: 1× SD of treated pre-treatment outcomes — a natural baseline
        # since RMSE exceeding natural variation indicates the synthetic control
        # fails to reproduce the treated series' level or trend.
        pre_treatment_sd = (
            np.std(Y_pre_treated_mean, ddof=1) if len(Y_pre_treated_mean) > 1 else 0.0
        )
        if pre_treatment_sd > 0 and pre_fit_rmse > pre_treatment_sd:
            warnings.warn(
                f"Pre-treatment fit is poor: RMSE ({pre_fit_rmse:.4f}) exceeds "
                f"the standard deviation of treated pre-treatment outcomes "
                f"({pre_treatment_sd:.4f}). The synthetic control may not "
                f"adequately reproduce treated unit trends. Consider adding "
                f"more control units or adjusting regularization.",
                UserWarning,
                stacklevel=2,
            )

        # Compute standard errors based on variance_method
        if self.variance_method == "bootstrap":
            se, bootstrap_estimates = self._bootstrap_se(
                Y_pre_control,
                Y_post_control,
                Y_pre_treated,
                Y_post_treated,
                unit_weights,
                time_weights,
                w_treated=w_treated,
                w_control=w_control,
                resolved_survey=_unit_resolved_survey,
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
                n_treated=len(treated_units),
                zeta_omega=zeta_omega,
                zeta_lambda=zeta_lambda,
                min_decrease=min_decrease,
                replications=self.n_bootstrap,
                w_control=w_control,
            )
            inference_method = "placebo"

        # Compute test statistics
        t_stat, p_value_analytical, conf_int = safe_inference(att, se, alpha=self.alpha)
        if len(placebo_effects) > 0 and np.isfinite(t_stat):
            p_value = max(
                np.mean(np.abs(placebo_effects) >= np.abs(att)),
                1.0 / (len(placebo_effects) + 1),
            )
        else:
            p_value = p_value_analytical

        # Create weight dictionaries.  When survey weights are active, store
        # the effective (composed) weights that were actually used for the ATT
        # so that results.unit_weights matches the estimator.
        unit_weights_dict = {unit_id: w for unit_id, w in zip(control_units, omega_eff)}
        time_weights_dict = {period: w for period, w in zip(pre_periods, time_weights)}

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
            noise_level=noise_level,
            zeta_omega=zeta_omega,
            zeta_lambda=zeta_lambda,
            pre_treatment_fit=pre_fit_rmse,
            placebo_effects=placebo_effects if len(placebo_effects) > 0 else None,
            n_bootstrap=self.n_bootstrap if inference_method == "bootstrap" else None,
            survey_metadata=survey_metadata,
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
        control_units: List[Any],
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
            Y_post_treated.astype(float),
        )

    def _residualize_covariates(
        self,
        data: pd.DataFrame,
        outcome: str,
        covariates: List[str],
        unit: str,
        time: str,
        survey_weights=None,
        survey_weight_type=None,
    ) -> pd.DataFrame:
        """
        Residualize outcome by regressing out covariates.

        Uses two-way fixed effects to partial out covariates. When survey
        weights are provided, uses WLS for population-representative
        covariate removal.
        """
        data = data.copy()

        # Create design matrix with covariates
        X = data[covariates].values.astype(float)

        # Add unit and time dummies
        unit_dummies = pd.get_dummies(data[unit], prefix="u", drop_first=True)
        time_dummies = pd.get_dummies(data[time], prefix="t", drop_first=True)

        X_full = np.column_stack([np.ones(len(data)), X, unit_dummies.values, time_dummies.values])

        y = data[outcome].values.astype(float)

        # Fit and get residuals using unified backend
        coeffs, residuals, _ = solve_ols(
            X_full,
            y,
            return_vcov=False,
            weights=survey_weights,
            weight_type=survey_weight_type,
        )

        # Add back the mean for interpretability
        if survey_weights is not None:
            y_center = np.average(y, weights=survey_weights)
        else:
            y_center = np.mean(y)
        data[outcome] = residuals + y_center

        return data

    @staticmethod
    def _build_unit_resolved_survey(data, unit_col, survey_design, control_units, treated_units):
        """Build a unit-level ResolvedSurveyDesign for Rao-Wu bootstrap.

        Extracts one row per unit (survey columns are unit-constant) in
        control-then-treated order matching the panel matrix columns.
        """
        from diff_diff.linalg import _factorize_cluster_ids
        from diff_diff.survey import ResolvedSurveyDesign

        all_units = list(control_units) + list(treated_units)
        # Take first row per unit in the specified order
        first_rows = data.groupby(unit_col).first().loc[all_units]
        n_units = len(all_units)

        # Weights (normalized pweights, mean=1)
        if survey_design.weights is not None:
            raw_w = first_rows[survey_design.weights].values.astype(np.float64)
            weights = raw_w * (n_units / np.sum(raw_w))
        else:
            weights = np.ones(n_units, dtype=np.float64)

        # Strata
        strata_arr = None
        n_strata = 0
        if survey_design.strata is not None:
            strata_arr = _factorize_cluster_ids(first_rows[survey_design.strata].values)
            n_strata = len(np.unique(strata_arr))

        # PSU
        psu_arr = None
        n_psu = 0
        if survey_design.psu is not None:
            psu_raw = first_rows[survey_design.psu].values
            if survey_design.nest and strata_arr is not None:
                combined = np.array([f"{s}_{p}" for s, p in zip(strata_arr, psu_raw)])
                psu_arr = _factorize_cluster_ids(combined)
            else:
                psu_arr = _factorize_cluster_ids(psu_raw)
            n_psu = len(np.unique(psu_arr))

        # FPC
        fpc_arr = None
        if survey_design.fpc is not None:
            fpc_arr = first_rows[survey_design.fpc].values.astype(np.float64)

        return ResolvedSurveyDesign(
            weights=weights,
            weight_type=survey_design.weight_type,
            strata=strata_arr,
            psu=psu_arr,
            fpc=fpc_arr,
            n_strata=n_strata,
            n_psu=n_psu,
            lonely_psu=survey_design.lonely_psu,
        )

    def _bootstrap_se(
        self,
        Y_pre_control: np.ndarray,
        Y_post_control: np.ndarray,
        Y_pre_treated: np.ndarray,
        Y_post_treated: np.ndarray,
        unit_weights: np.ndarray,
        time_weights: np.ndarray,
        w_treated=None,
        w_control=None,
        resolved_survey=None,
    ) -> Tuple[float, np.ndarray]:
        """Compute bootstrap standard error matching R's synthdid bootstrap_sample.

        Resamples all units (control + treated) with replacement, renormalizes
        original unit weights for the resampled controls, and computes the
        SDID estimator with **fixed** weights (no re-estimation).

        When ``resolved_survey`` is provided (unit-level ResolvedSurveyDesign
        with strata/PSU/FPC), uses Rao-Wu rescaled bootstrap instead of the
        simple pairs bootstrap.  The Rao-Wu weights are per-unit rescaled
        survey weights; they composite with SDID unit weights the same way
        pweights do in the weights-only path.

        This matches R's ``synthdid::vcov(method="bootstrap")``.
        """
        from diff_diff.bootstrap_utils import generate_rao_wu_weights

        rng = np.random.default_rng(self.seed)
        n_control = Y_pre_control.shape[1]
        n_treated = Y_pre_treated.shape[1]
        n_total = n_control + n_treated

        # Build full panel matrix: (n_pre+n_post, n_control+n_treated)
        Y_full = np.block([[Y_pre_control, Y_pre_treated], [Y_post_control, Y_post_treated]])
        n_pre = Y_pre_control.shape[0]

        # Determine whether to use Rao-Wu (full design) or pairs bootstrap
        _use_rao_wu = resolved_survey is not None

        # Check for unidentified variance (single unstratified PSU)
        if (
            _use_rao_wu
            and resolved_survey.psu is not None
            and resolved_survey.n_psu < 2
            and resolved_survey.strata is None
        ):
            return np.nan, np.array([])

        bootstrap_estimates = []

        for _ in range(self.n_bootstrap):
            if _use_rao_wu:
                # --- Rao-Wu rescaled bootstrap path ---
                # generate_rao_wu_weights returns per-unit rescaled survey
                # weights (shape n_total).  Units whose PSU was not drawn
                # get weight 0, effectively dropping them.
                try:
                    boot_rw = generate_rao_wu_weights(resolved_survey, rng)

                    rw_control = boot_rw[:n_control]
                    rw_treated = boot_rw[n_control:]

                    # Skip if all control or all treated weights are zero
                    if rw_control.sum() == 0 or rw_treated.sum() == 0:
                        continue

                    # Composite SDID unit weights with Rao-Wu rescaled weights
                    boot_omega_eff = unit_weights * rw_control
                    if boot_omega_eff.sum() > 0:
                        boot_omega_eff = boot_omega_eff / boot_omega_eff.sum()
                    else:
                        continue

                    # Treated mean weighted by Rao-Wu weights
                    Y_boot_pre_t_mean = np.average(
                        Y_pre_treated,
                        axis=1,
                        weights=rw_treated,
                    )
                    Y_boot_post_t_mean = np.average(
                        Y_post_treated,
                        axis=1,
                        weights=rw_treated,
                    )

                    tau = compute_sdid_estimator(
                        Y_pre_control,
                        Y_post_control,
                        Y_boot_pre_t_mean,
                        Y_boot_post_t_mean,
                        boot_omega_eff,
                        time_weights,
                    )
                    if np.isfinite(tau):
                        bootstrap_estimates.append(tau)

                except (ValueError, LinAlgError):
                    continue
            else:
                # --- Standard pairs bootstrap path (weights-only or no survey) ---
                # Resample ALL units with replacement
                boot_idx = rng.choice(n_total, size=n_total, replace=True)

                # Identify which resampled units are control vs treated
                boot_is_control = boot_idx < n_control
                boot_control_idx = boot_idx[boot_is_control]
                boot_treated_idx = boot_idx[~boot_is_control]

                # Skip if no control or no treated units in bootstrap sample
                if len(boot_control_idx) == 0 or len(boot_treated_idx) == 0:
                    continue

                try:
                    # Renormalize original unit weights for the resampled controls
                    boot_omega = _sum_normalize(unit_weights[boot_control_idx])

                    # Compose with control survey weights if present
                    if w_control is not None:
                        boot_w_c = w_control[boot_idx[boot_is_control]]
                        boot_omega_eff = boot_omega * boot_w_c
                        boot_omega_eff = boot_omega_eff / boot_omega_eff.sum()
                    else:
                        boot_omega_eff = boot_omega

                    # Extract resampled outcome matrices
                    Y_boot = Y_full[:, boot_idx]
                    Y_boot_pre_c = Y_boot[:n_pre, boot_is_control]
                    Y_boot_post_c = Y_boot[n_pre:, boot_is_control]
                    Y_boot_pre_t = Y_boot[:n_pre, ~boot_is_control]
                    Y_boot_post_t = Y_boot[n_pre:, ~boot_is_control]

                    # Compute ATT with FIXED weights (do NOT re-estimate).
                    # boot_idx[~boot_is_control] maps to original index space;
                    # subtract n_control to index into w_treated. Duplicate draws
                    # carry identical weights -> alignment is safe.
                    if w_treated is not None:
                        boot_w_t = w_treated[boot_idx[~boot_is_control] - n_control]
                        Y_boot_pre_t_mean = np.average(
                            Y_boot_pre_t,
                            axis=1,
                            weights=boot_w_t,
                        )
                        Y_boot_post_t_mean = np.average(
                            Y_boot_post_t,
                            axis=1,
                            weights=boot_w_t,
                        )
                    else:
                        Y_boot_pre_t_mean = np.mean(Y_boot_pre_t, axis=1)
                        Y_boot_post_t_mean = np.mean(Y_boot_post_t, axis=1)

                    tau = compute_sdid_estimator(
                        Y_boot_pre_c,
                        Y_boot_post_c,
                        Y_boot_pre_t_mean,
                        Y_boot_post_t_mean,
                        boot_omega_eff,
                        time_weights,
                    )
                    if np.isfinite(tau):
                        bootstrap_estimates.append(tau)

                except (ValueError, LinAlgError):
                    continue

        bootstrap_estimates = np.array(bootstrap_estimates)

        # Check bootstrap success rate and handle failures
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
                f"Consider using variance_method='placebo' or increasing "
                f"the regularization parameters (zeta_omega, zeta_lambda)."
            )
        elif n_successful == 1:
            warnings.warn(
                f"Only 1/{self.n_bootstrap} bootstrap iteration succeeded. "
                f"Standard error cannot be computed reliably (requires at least 2). "
                f"Returning SE=0.0. Consider using variance_method='placebo' or "
                f"increasing the regularization (zeta_omega, zeta_lambda).",
                UserWarning,
                stacklevel=2,
            )
            se = 0.0
        elif failure_rate > 0.05:
            warnings.warn(
                f"Only {n_successful}/{self.n_bootstrap} bootstrap iterations succeeded "
                f"({failure_rate:.1%} failure rate). Standard errors may be unreliable. "
                f"This can occur with small samples or insufficient pre-treatment periods.",
                UserWarning,
                stacklevel=2,
            )
            se = float(np.std(bootstrap_estimates, ddof=1))
        else:
            se = float(np.std(bootstrap_estimates, ddof=1))

        return se, bootstrap_estimates

    def _placebo_variance_se(
        self,
        Y_pre_control: np.ndarray,
        Y_post_control: np.ndarray,
        Y_pre_treated_mean: np.ndarray,
        Y_post_treated_mean: np.ndarray,
        n_treated: int,
        zeta_omega: float = 0.0,
        zeta_lambda: float = 0.0,
        min_decrease: float = 1e-5,
        replications: int = 200,
        w_control=None,
    ) -> Tuple[float, np.ndarray]:
        """
        Compute placebo-based variance matching R's synthdid methodology.

        This implements Algorithm 4 from Arkhangelsky et al. (2021),
        matching R's synthdid::vcov(method = "placebo"):

        1. Randomly sample N₀ control indices (permutation)
        2. Designate last N₁ as pseudo-treated, first (N₀-N₁) as pseudo-controls
        3. Re-estimate both omega and lambda on the permuted data (from
           uniform initialization, fresh start), matching R's behavior where
           ``update.omega=TRUE, update.lambda=TRUE`` are passed via ``opts``
        4. Compute SDID estimate with re-estimated weights
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
        n_treated : int
            Number of treated units in the original estimation.
        zeta_omega : float
            Regularization parameter for unit weights (for re-estimation).
        zeta_lambda : float
            Regularization parameter for time weights (for re-estimation).
        min_decrease : float
            Convergence threshold for Frank-Wolfe (for re-estimation).
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

                # Get pseudo-control and pseudo-treated outcomes
                Y_pre_pseudo_control = Y_pre_control[:, pseudo_control_idx]
                Y_post_pseudo_control = Y_post_control[:, pseudo_control_idx]

                # Pseudo-treated means: survey-weighted when available
                if w_control is not None:
                    pseudo_w_tr = w_control[pseudo_treated_idx]
                    Y_pre_pseudo_treated_mean = np.average(
                        Y_pre_control[:, pseudo_treated_idx],
                        axis=1,
                        weights=pseudo_w_tr,
                    )
                    Y_post_pseudo_treated_mean = np.average(
                        Y_post_control[:, pseudo_treated_idx],
                        axis=1,
                        weights=pseudo_w_tr,
                    )
                else:
                    Y_pre_pseudo_treated_mean = np.mean(
                        Y_pre_control[:, pseudo_treated_idx], axis=1
                    )
                    Y_post_pseudo_treated_mean = np.mean(
                        Y_post_control[:, pseudo_treated_idx], axis=1
                    )

                # Re-estimate weights on permuted data (matching R's behavior)
                # R passes update.omega=TRUE, update.lambda=TRUE via opts,
                # re-estimating weights from uniform initialization (fresh start).
                # Unit weights: re-estimate on pseudo-control/pseudo-treated data
                pseudo_omega = compute_sdid_unit_weights(
                    Y_pre_pseudo_control,
                    Y_pre_pseudo_treated_mean,
                    zeta_omega=zeta_omega,
                    min_decrease=min_decrease,
                )

                # Compose pseudo_omega with control survey weights
                if w_control is not None:
                    pseudo_w_co = w_control[pseudo_control_idx]
                    pseudo_omega_eff = pseudo_omega * pseudo_w_co
                    pseudo_omega_eff = pseudo_omega_eff / pseudo_omega_eff.sum()
                else:
                    pseudo_omega_eff = pseudo_omega

                # Time weights: re-estimate on pseudo-control data
                pseudo_lambda = compute_time_weights(
                    Y_pre_pseudo_control,
                    Y_post_pseudo_control,
                    zeta_lambda=zeta_lambda,
                    min_decrease=min_decrease,
                )

                # Compute placebo SDID estimate (step 4)
                tau = compute_sdid_estimator(
                    Y_pre_pseudo_control,
                    Y_post_pseudo_control,
                    Y_pre_pseudo_treated_mean,
                    Y_post_pseudo_treated_mean,
                    pseudo_omega_eff,
                    pseudo_lambda,
                )
                if np.isfinite(tau):
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
        se = np.sqrt((n_successful - 1) / n_successful) * np.std(placebo_estimates, ddof=1)

        return se, placebo_estimates

    def get_params(self) -> Dict[str, Any]:
        """Get estimator parameters."""
        return {
            "zeta_omega": self.zeta_omega,
            "zeta_lambda": self.zeta_lambda,
            "alpha": self.alpha,
            "variance_method": self.variance_method,
            "n_bootstrap": self.n_bootstrap,
            "seed": self.seed,
        }

    def set_params(self, **params) -> "SyntheticDiD":
        """Set estimator parameters."""
        # Deprecated parameter names — emit warning and ignore
        _deprecated = {"lambda_reg", "zeta"}
        for key, value in params.items():
            if key in _deprecated:
                warnings.warn(
                    f"{key} is deprecated and ignored. Use zeta_omega/zeta_lambda " f"instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            elif hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")
        return self
