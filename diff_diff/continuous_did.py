"""
Continuous Difference-in-Differences estimator.

Implements Callaway, Goodman-Bacon & Sant'Anna (2024),
"Difference-in-Differences with a Continuous Treatment" (NBER WP 32117).

Estimates dose-response curves ATT(d) and ACRT(d), as well as summary
parameters ATT^{glob} and ACRT^{glob}, with optional multiplier bootstrap
inference.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from diff_diff.bootstrap_utils import (
    compute_effect_bootstrap_stats,
    generate_bootstrap_weights_batch,
)
from diff_diff.continuous_did_bspline import (
    bspline_derivative_design_matrix,
    bspline_design_matrix,
    build_bspline_basis,
    default_dose_grid,
)
from diff_diff.continuous_did_results import (
    ContinuousDiDResults,
    DoseResponseCurve,
)
from diff_diff.linalg import _rank_guarded_inv, solve_logit, solve_ols
from diff_diff.survey import (
    _resolve_survey_for_fit,
    _validate_unit_constant_survey,
    build_unit_first_row_index,
    compute_survey_vcov,
)
from diff_diff.utils import safe_inference

__all__ = ["ContinuousDiD", "ContinuousDiDResults", "DoseResponseCurve"]


class ContinuousDiD:
    """
    Continuous Difference-in-Differences estimator.

    Implements the methodology from Callaway, Goodman-Bacon & Sant'Anna (2024)
    for estimating dose-response curves when treatment has a continuous intensity.

    Parameters
    ----------
    degree : int, default=3
        B-spline degree (3 = cubic).
    num_knots : int, default=0
        Number of interior knots for the B-spline basis.
    dvals : array-like, optional
        Custom dose evaluation grid. If None, uses quantile-based default.
    control_group : str, default="never_treated"
        ``"never_treated"`` or ``"not_yet_treated"``.
    anticipation : int, default=0
        Number of periods of treatment anticipation.
    base_period : str, default="varying"
        ``"varying"`` or ``"universal"``.
    alpha : float, default=0.05
        Significance level for confidence intervals.
    n_bootstrap : int, default=0
        Number of multiplier bootstrap iterations. 0 for analytical SEs only.
    bootstrap_weights : str, default="rademacher"
        Bootstrap weight type: ``"rademacher"``, ``"mammen"``, or ``"webb"``.
    seed : int, optional
        Random seed for reproducibility.
    rank_deficient_action : str, default="warn"
        Action for rank-deficient B-spline OLS: ``"warn"``, ``"error"``, or ``"silent"``.
    covariates : list of str, optional
        Column names of covariates for **conditional** parallel trends
        (``E[ΔY(0) | D=d, X] = E[ΔY(0) | D=0, X]``). When ``None`` (default) the
        estimator uses unconditional parallel trends. Covariates enter through a
        covariate-adjusted per-cell control counterfactual (see ``estimation_method``).
        Covariates are read from the base period of each ``(g, t)`` comparison.
        Not currently composable with ``survey_design=`` (raises ``NotImplementedError``).
    estimation_method : str, default="dr"
        Covariate-adjustment method (only used when ``covariates`` is set):
        ``"reg"`` (outcome regression) or ``"dr"`` (doubly-robust, the default).
        ``"ipw"`` is **not supported on the dose / event-study aggregation** — pure
        IPW's covariate adjustment is a single scalar level shift, so it cannot
        adjust the dose-response *shape* (ACRT(d) would be identical to the
        unconditional fit); it raises ``NotImplementedError``. ``reg`` and ``dr``
        share the dose-response shape and ACRT(d); ``dr`` differs only in the
        ``overall_att`` / ATT(d) level and in its doubly-robust standard errors.
    pscore_trim : float, default=0.01
        Propensity-score trimming bound for the ``dr`` path (scores clipped to
        ``[pscore_trim, 1 - pscore_trim]``).
    epv_threshold : float, default=10.0
        Events-per-variable threshold for the ``dr`` propensity logit diagnostics.
    pscore_fallback : str, default="error"
        Action when ``dr`` propensity estimation raises (the logit IRLS fails
        with a ``LinAlgError`` / ``ValueError``, e.g. perfect separation or rank
        deficiency): ``"error"`` (re-raise — the default, fail-closed so a `dr`
        fit never silently degrades to a non-DR estimate) or ``"unconditional"``
        (fall back to an unconditional propensity with a warning; the affected
        cells are then reg-like — use only when you knowingly accept that). Note:
        low events-per-variable emits a diagnostic warning but does not itself
        trigger the fallback.

    Examples
    --------
    >>> from diff_diff import ContinuousDiD, generate_continuous_did_data
    >>> data = generate_continuous_did_data(n_units=200, seed=42)
    >>> est = ContinuousDiD(n_bootstrap=199, seed=42)
    >>> results = est.fit(data, outcome="outcome", unit="unit",
    ...                   time="period", first_treat="first_treat",
    ...                   dose="dose", aggregate="dose")
    >>> results.overall_att  # doctest: +SKIP
    """

    _VALID_CONTROL_GROUPS = {"never_treated", "not_yet_treated"}
    _VALID_BASE_PERIODS = {"varying", "universal"}
    _VALID_ESTIMATION_METHODS = {"reg", "dr", "ipw"}

    def __init__(
        self,
        degree: int = 3,
        num_knots: int = 0,
        dvals: Optional[np.ndarray] = None,
        control_group: str = "never_treated",
        anticipation: int = 0,
        base_period: str = "varying",
        alpha: float = 0.05,
        n_bootstrap: int = 0,
        bootstrap_weights: str = "rademacher",
        seed: Optional[int] = None,
        rank_deficient_action: str = "warn",
        covariates: Optional[List[str]] = None,
        estimation_method: str = "dr",
        pscore_trim: float = 0.01,
        epv_threshold: float = 10.0,
        pscore_fallback: str = "error",
    ):
        self.degree = degree
        self.num_knots = num_knots
        self.dvals = np.asarray(dvals, dtype=float) if dvals is not None else None
        self.control_group = control_group
        self.anticipation = anticipation
        self.base_period = base_period
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap
        self.bootstrap_weights = bootstrap_weights
        self.seed = seed
        self.rank_deficient_action = rank_deficient_action
        self.covariates = covariates
        self.estimation_method = estimation_method
        self.pscore_trim = pscore_trim
        self.epv_threshold = epv_threshold
        self.pscore_fallback = pscore_fallback
        self._validate_constrained_params()

    def _validate_constrained_params(self) -> None:
        """Validate control_group, base_period, and estimation_method values."""
        if self.control_group not in self._VALID_CONTROL_GROUPS:
            raise ValueError(
                f"Invalid control_group: '{self.control_group}'. "
                f"Must be one of {self._VALID_CONTROL_GROUPS}."
            )
        if self.base_period not in self._VALID_BASE_PERIODS:
            raise ValueError(
                f"Invalid base_period: '{self.base_period}'. "
                f"Must be one of {self._VALID_BASE_PERIODS}."
            )
        if self.estimation_method not in self._VALID_ESTIMATION_METHODS:
            raise ValueError(
                f"Invalid estimation_method: '{self.estimation_method}'. "
                f"Must be one of {self._VALID_ESTIMATION_METHODS}."
            )
        if self.pscore_fallback not in {"unconditional", "error"}:
            raise ValueError(
                f"Invalid pscore_fallback: '{self.pscore_fallback}'. "
                "Must be 'unconditional' or 'error'."
            )
        if not (np.isfinite(self.pscore_trim) and 0.0 <= self.pscore_trim < 0.5):
            raise ValueError(
                f"Invalid pscore_trim: {self.pscore_trim}. " "Must be finite and in [0, 0.5)."
            )
        if not (np.isfinite(self.epv_threshold) and self.epv_threshold > 0):
            raise ValueError(
                f"Invalid epv_threshold: {self.epv_threshold}. Must be finite and > 0."
            )

    def get_params(self) -> Dict[str, Any]:
        """Return estimator parameters as a dictionary."""
        return {
            "degree": self.degree,
            "num_knots": self.num_knots,
            "dvals": self.dvals,
            "control_group": self.control_group,
            "anticipation": self.anticipation,
            "base_period": self.base_period,
            "alpha": self.alpha,
            "n_bootstrap": self.n_bootstrap,
            "bootstrap_weights": self.bootstrap_weights,
            "seed": self.seed,
            "rank_deficient_action": self.rank_deficient_action,
            "covariates": self.covariates,
            "estimation_method": self.estimation_method,
            "pscore_trim": self.pscore_trim,
            "epv_threshold": self.epv_threshold,
            "pscore_fallback": self.pscore_fallback,
        }

    def set_params(self, **params) -> "ContinuousDiD":
        """Set estimator parameters and return self.

        Transactional: the candidate configuration is validated against a clone
        before any attribute on ``self`` is mutated, so an invalid update leaves
        the estimator unchanged (no half-applied state).
        """
        for key in params:
            if not hasattr(self, key):
                raise ValueError(f"Invalid parameter: {key}")
        # Validate the candidate config on a throwaway before mutating self.
        candidate = ContinuousDiD(**{**self.get_params(), **params})
        # candidate.__init__ ran _validate_constrained_params() successfully.
        del candidate
        for key, value in params.items():
            setattr(self, key, value)
        return self

    # ------------------------------------------------------------------
    # Main fit
    # ------------------------------------------------------------------

    def fit(
        self,
        data: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        first_treat: str,
        dose: str,
        aggregate: Optional[str] = None,
        survey_design: object = None,
    ) -> ContinuousDiDResults:
        """
        Fit the continuous DiD estimator.

        Parameters
        ----------
        data : pd.DataFrame
            Panel data.
        outcome : str
            Outcome column name.
        unit : str
            Unit identifier column.
        time : str
            Time period column.
        first_treat : str
            First treatment period column (0 or inf for never-treated).
        dose : str
            Continuous dose column.
        aggregate : str, optional
            ``"dose"`` for dose-response aggregation, ``"eventstudy"`` for
            binarized event study.
        survey_design : SurveyDesign, optional
            Survey design specification for design-based inference.
            Supports weighted estimation and Taylor series linearization
            variance with strata, PSU, and FPC.

        Returns
        -------
        ContinuousDiDResults
        """
        # 1. Validate & prepare
        _VALID_AGGREGATES = (None, "dose", "eventstudy")
        if aggregate not in _VALID_AGGREGATES:
            raise ValueError(
                f"Invalid aggregate: '{aggregate}'. " f"Must be one of {_VALID_AGGREGATES}."
            )

        # Resolve survey design if provided
        resolved_survey, survey_weights, survey_weight_type, survey_metadata = (
            _resolve_survey_for_fit(survey_design, data, "analytical")
        )

        # Validate within-unit constancy for panel survey designs
        if resolved_survey is not None:
            _validate_unit_constant_survey(data, unit, survey_design)

        # Bootstrap + survey supported via PSU-level multiplier bootstrap.

        df = data.copy()
        cov_cols = list(self.covariates) if self.covariates else []
        for col in [outcome, unit, time, first_treat, dose, *cov_cols]:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in data.")

        # Covariate-path guards (conditional parallel trends).
        if cov_cols:
            if survey_design is not None:
                raise NotImplementedError(
                    "ContinuousDiD does not yet support covariates= together with "
                    "survey_design= (weighted covariate outcome-regression / "
                    "propensity influence functions are a follow-up). Use one or "
                    "the other for now."
                )
            if self.estimation_method == "ipw":
                raise NotImplementedError(
                    "estimation_method='ipw' is not supported with covariates on the "
                    "dose-response / event-study aggregation. Pure IPW's covariate "
                    "adjustment is a single scalar (a propensity-reweighted control "
                    "mean), which shifts only the ATT(d) level and leaves ACRT(d) "
                    "identical to the unconditional fit — it cannot adjust the "
                    "dose-response shape. Use estimation_method='reg' or 'dr'."
                )
            # Fail closed on missing/non-finite covariates: a per-cell fallback to
            # unconditional estimation would silently mix conditional-PT and
            # unconditional-PT cells in the aggregate (no-silent-failures).
            cov_nonfinite = ~np.isfinite(df[cov_cols].to_numpy(dtype=float))
            if cov_nonfinite.any():
                n_bad = int(cov_nonfinite.any(axis=1).sum())
                raise ValueError(
                    f"{n_bad} row(s) have missing/non-finite covariate values. "
                    "ContinuousDiD requires complete covariates (a per-cell fallback "
                    "would mix conditional and unconditional estimands). Drop or "
                    "impute the affected rows, or fit without covariates."
                )

        # Verify dose is time-invariant
        dose_nunique = df.groupby(unit)[dose].nunique()
        if dose_nunique.max() > 1:
            bad_units = dose_nunique[dose_nunique > 1].index.tolist()
            raise ValueError(
                f"Dose must be time-invariant. Units with varying dose: {bad_units[:5]}"
            )

        # Normalize first_treat: +inf → 0 (R-style never-treated encoding).
        # Count rows recategorized so users can see how many units just
        # crossed from "treated at some point" to "never treated" — silent
        # recategorization here would shift the control composition (axis-E
        # silent coercion). Only positive infinity is recoded (to match the
        # existing `.replace([np.inf, float("inf")], 0)` semantics on the
        # next line).
        first_treat_vals = df[first_treat].values
        # Reject NaN first_treat explicitly. NaN survives preprocessing but
        # satisfies neither the treated (g > 0) nor never-treated (g == 0)
        # mask, so affected units would be silently excluded from the
        # estimator (same silent-failure shape as `first_treat < 0`).
        nan_mask = pd.isna(df[first_treat])
        n_nan_first_treat = int(nan_mask.sum())
        if n_nan_first_treat > 0:
            raise ValueError(
                f"{n_nan_first_treat} row(s) have NaN '{first_treat}' "
                f"values. Valid values are 0 (never-treated) or a positive "
                f"treatment period; such units would otherwise be silently "
                f"excluded from both treated and control pools."
            )
        inf_mask = np.isposinf(first_treat_vals)
        n_inf_first_treat = int(inf_mask.sum())
        if n_inf_first_treat > 0:
            warnings.warn(
                f"{n_inf_first_treat} row(s) have inf in '{first_treat}'; "
                f"treating the corresponding units as never-treated. Pass an "
                f"explicit never-treated marker (0) if this is not intended.",
                UserWarning,
                stacklevel=2,
            )
        # Reject negative first_treat values (including -inf) explicitly.
        # Without this guard they would survive preprocessing but fall out of
        # both the treated (g > 0) and never-treated (g == 0) masks, silently
        # excluding the affected units.
        negative_mask = first_treat_vals < 0
        n_negative_first_treat = int(negative_mask.sum())
        if n_negative_first_treat > 0:
            raise ValueError(
                f"{n_negative_first_treat} row(s) have negative '{first_treat}' "
                f"values (including -inf). Valid values are 0 (never-treated) "
                f"or a positive treatment period; such units would otherwise "
                f"be silently excluded from both treated and control pools."
            )
        df[first_treat] = df[first_treat].replace([np.inf, float("inf")], 0)

        # Drop units with positive first_treat but zero dose (R convention)
        unit_info = df.groupby(unit).first()[[first_treat, dose]]
        drop_units = unit_info[(unit_info[first_treat] > 0) & (unit_info[dose] == 0)].index
        if len(drop_units) > 0:
            warnings.warn(
                f"Dropping {len(drop_units)} units with positive first_treat but zero dose.",
                UserWarning,
                stacklevel=2,
            )
            df = df[~df[unit].isin(drop_units)]

        # Validate no negative doses among treated units
        treated_doses = df.loc[df[first_treat] > 0, dose]
        if (treated_doses < 0).any():
            n_neg = int((treated_doses < 0).sum())
            raise ValueError(
                f"Found {n_neg} treated unit(s) with negative dose. "
                f"Dose must be strictly positive for treated units (D > 0)."
            )

        # Detect discrete (integer-valued) dose among treated units
        unit_doses = df.loc[df[first_treat] > 0].groupby(unit)[dose].first()
        unique_pos_doses = unit_doses[unit_doses > 0].unique()
        is_integer = len(unique_pos_doses) > 0 and np.allclose(
            unique_pos_doses, np.round(unique_pos_doses)
        )
        if is_integer:
            warnings.warn(
                f"Dose appears discrete ({len(unique_pos_doses)} unique integer values). "
                "B-spline smoothing may be inappropriate for discrete treatments. "
                "Consider a saturated regression approach (not yet implemented).",
                UserWarning,
                stacklevel=2,
            )

        # Force dose=0 for never-treated units with nonzero dose. Report the
        # affected row count via UserWarning so users can see whether their
        # never-treated rows had unintended nonzero doses — silent zeroing
        # here would quietly shift part of the control trajectory (axis-E
        # silent coercion, paired with the `first_treat=inf -> 0` fix above).
        never_treated_mask = df[first_treat] == 0
        nonzero_dose_rows = never_treated_mask & (df[dose] != 0)
        n_nonzero_dose_never_treated = int(nonzero_dose_rows.sum())
        if n_nonzero_dose_never_treated > 0:
            warnings.warn(
                f"{n_nonzero_dose_never_treated} row(s) have '{first_treat}'=0 "
                f"(never-treated) but nonzero '{dose}'; zeroing the dose. Pass "
                f"dose=0 for never-treated rows to avoid this coercion.",
                UserWarning,
                stacklevel=2,
            )
            df.loc[never_treated_mask, dose] = 0.0

        # Verify balanced panel
        all_periods = set(df[time].unique())
        unit_periods = df.groupby(unit)[time].apply(set)
        is_unbalanced = unit_periods.apply(lambda s: s != all_periods)
        if is_unbalanced.any():
            n_bad = int(is_unbalanced.sum())
            raise ValueError(
                "Unbalanced panel detected. ContinuousDiD requires a balanced panel. "
                f"{n_bad} unit(s) have missing periods."
            )

        # Identify groups and time periods
        unit_cohort = df.groupby(unit)[first_treat].first()
        treatment_groups = sorted([g for g in unit_cohort.unique() if g > 0])
        time_periods = sorted(df[time].unique())

        if len(treatment_groups) == 0:
            raise ValueError("No treated units found (all first_treat == 0).")

        n_control = int((unit_cohort == 0).sum())
        if self.control_group == "never_treated" and n_control == 0:
            raise ValueError(
                "No never-treated units found. Use control_group='not_yet_treated' "
                "or add never-treated units."
            )

        if self.control_group == "not_yet_treated" and n_control == 0:
            raise ValueError(
                "No never-treated (D=0) units found. With control_group='not_yet_treated', "
                "dose-response curve identification requires P(D=0) > 0 "
                "(Remark 3.1 in Callaway et al. is not yet implemented). "
                "Add never-treated units or use a dataset with D=0 observations."
            )

        # Re-resolve survey design on filtered df if rows were dropped
        # (survey arrays must align with df, not the original data)
        if resolved_survey is not None and len(df) < len(data):
            resolved_survey, survey_weights, survey_weight_type, survey_metadata = (
                _resolve_survey_for_fit(survey_design, df, "analytical")
            )

        # 2. Precompute structures
        precomp = self._precompute_structures(
            df,
            outcome,
            unit,
            time,
            first_treat,
            dose,
            time_periods,
            survey_weights=survey_weights,
            covariates=self.covariates,
        )

        # Compute dvals (evaluation grid)
        all_treated_doses = precomp["dose_vector"][precomp["dose_vector"] > 0]
        if self.dvals is not None:
            dvals = self.dvals
        else:
            dvals = default_dose_grid(all_treated_doses)

        # Build B-spline knots from all treated doses
        knots, degree = build_bspline_basis(
            all_treated_doses, degree=self.degree, num_knots=self.num_knots
        )

        # 3. Iterate over (g,t) cells
        gt_results = {}
        gt_bootstrap_info = {}

        for g in treatment_groups:
            for t in time_periods:
                result = self._compute_dose_response_gt(
                    precomp,
                    g,
                    t,
                    knots,
                    degree,
                    dvals,
                    survey_weights=precomp.get("unit_survey_weights"),
                    resolved_survey=resolved_survey,
                )
                if result is not None:
                    gt_results[(g, t)] = result
                    gt_bootstrap_info[(g, t)] = result.get("_bootstrap_info", {})

        # Filter out NaN cells (e.g., from zero effective survey mass)
        gt_results = {
            gt: r for gt, r in gt_results.items() if np.isfinite(r.get("att_glob", np.nan))
        }

        if len(gt_results) == 0:
            raise ValueError("No valid (g,t) cells computed.")

        # 4. Aggregate
        post_gt = {(g, t): r for (g, t), r in gt_results.items() if t >= g - self.anticipation}

        # Dose-response aggregation
        n_grid = len(dvals)

        # NaN-initialized SE/CI fields (used when post_gt is empty or as defaults)
        att_d_se = np.full(n_grid, np.nan)
        att_d_ci_lower = np.full(n_grid, np.nan)
        att_d_ci_upper = np.full(n_grid, np.nan)
        acrt_d_se = np.full(n_grid, np.nan)
        acrt_d_ci_lower = np.full(n_grid, np.nan)
        acrt_d_ci_upper = np.full(n_grid, np.nan)
        overall_att_se = np.nan
        overall_att_t = np.nan
        overall_att_p = np.nan
        overall_att_ci = (np.nan, np.nan)
        overall_acrt_se = np.nan
        overall_acrt_t = np.nan
        overall_acrt_p = np.nan
        overall_acrt_ci = (np.nan, np.nan)
        att_d_p = None
        acrt_d_p = None

        # Event study aggregation (binarized) — runs on ALL (g,t) cells
        event_study_effects = None
        if aggregate == "eventstudy":
            event_study_effects = self._aggregate_event_study(
                gt_results,
                gt_bootstrap_info=gt_bootstrap_info,
                unit_survey_weights=precomp.get("unit_survey_weights"),
                unit_cohorts=precomp["unit_cohorts"],
                anticipation=self.anticipation,
            )

        _survey_df = None  # Set by analytical branch when survey is active

        if len(post_gt) == 0:
            warnings.warn(
                "No post-treatment (g,t) cells available for aggregation. "
                "This can occur when all treatments start after the last observed "
                "period or all cells were skipped due to insufficient data.",
                UserWarning,
                stacklevel=2,
            )
            overall_att = np.nan
            overall_acrt = np.nan
            agg_att_d = np.full(n_grid, np.nan)
            agg_acrt_d = np.full(n_grid, np.nan)
        else:
            # Compute cell weights: group-proportional (matching R's contdid convention).
            # Each group g gets weight proportional to its number of treated units.
            # When survey weights present, use sum(w_g) / sum(w) instead of n_g / N.
            # Within each group, weight is divided equally among post-treatment cells.
            group_n_treated = {}
            group_n_post_cells = {}
            unit_sw = precomp.get("unit_survey_weights")
            for (g, t), r in post_gt.items():
                if g not in group_n_treated:
                    if unit_sw is not None:
                        # Survey-weighted group size: sum of weights for treated units in g
                        g_mask = precomp["unit_cohorts"] == g
                        group_n_treated[g] = float(np.sum(unit_sw[g_mask]))
                    else:
                        group_n_treated[g] = float(r["n_treated"])
                    group_n_post_cells[g] = 0
                group_n_post_cells[g] += 1

            total_treated = sum(group_n_treated.values())
            cell_weights = {}
            if total_treated > 0:
                for (g, t), r in post_gt.items():
                    pg = group_n_treated[g] / total_treated
                    cell_weights[(g, t)] = pg / group_n_post_cells[g]

            agg_att_d = np.zeros(n_grid)
            agg_acrt_d = np.zeros(n_grid)
            overall_att = 0.0
            overall_acrt = 0.0

            for gt, w in cell_weights.items():
                r = post_gt[gt]
                agg_att_d += w * r["att_d"]
                agg_acrt_d += w * r["acrt_d"]
                overall_att += w * r["att_glob"]
                overall_acrt += w * r["acrt_glob"]

            # 5. Bootstrap / Analytical SE
            if self.n_bootstrap > 0:
                boot_result = self._run_bootstrap(
                    precomp,
                    gt_results,
                    gt_bootstrap_info,
                    post_gt,
                    cell_weights,
                    knots,
                    degree,
                    dvals,
                    overall_att,
                    overall_acrt,
                    agg_att_d,
                    agg_acrt_d,
                    event_study_effects,
                    resolved_survey=resolved_survey,
                )
                att_d_se = boot_result["att_d_se"]
                att_d_ci_lower = boot_result["att_d_ci_lower"]
                att_d_ci_upper = boot_result["att_d_ci_upper"]
                acrt_d_se = boot_result["acrt_d_se"]
                acrt_d_ci_lower = boot_result["acrt_d_ci_lower"]
                acrt_d_ci_upper = boot_result["acrt_d_ci_upper"]
                att_d_p = boot_result["att_d_p"]
                acrt_d_p = boot_result["acrt_d_p"]
                overall_att_se = boot_result["overall_att_se"]
                overall_att_t = safe_inference(overall_att, overall_att_se, self.alpha)[0]
                overall_att_p = boot_result["overall_att_p"]
                overall_att_ci = boot_result["overall_att_ci"]
                overall_acrt_se = boot_result["overall_acrt_se"]
                overall_acrt_t = safe_inference(overall_acrt, overall_acrt_se, self.alpha)[0]
                overall_acrt_p = boot_result["overall_acrt_p"]
                overall_acrt_ci = boot_result["overall_acrt_ci"]
                if event_study_effects is not None:
                    for e, info in event_study_effects.items():
                        if e in boot_result.get("es_se", {}):
                            info["se"] = boot_result["es_se"][e]
                            info["t_stat"] = safe_inference(info["effect"], info["se"], self.alpha)[
                                0
                            ]
                            info["p_value"] = boot_result["es_p"][e]
                            info["conf_int"] = boot_result["es_ci"][e]
            else:
                # Analytical SEs via influence functions
                analytic = self._compute_analytical_se(
                    precomp,
                    gt_results,
                    gt_bootstrap_info,
                    post_gt,
                    cell_weights,
                    knots,
                    degree,
                    dvals,
                    agg_att_d,
                    agg_acrt_d,
                    resolved_survey=resolved_survey,
                )
                att_d_se = analytic["att_d_se"]
                acrt_d_se = analytic["acrt_d_se"]
                overall_att_se = analytic["overall_att_se"]
                overall_acrt_se = analytic["overall_acrt_se"]

                # Survey df for t-distribution inference (unit-level, not panel-level)
                _survey_df = analytic.get("df_survey")
                # Guard: replicate design with undefined df → NaN inference
                if (
                    _survey_df is None
                    and resolved_survey is not None
                    and hasattr(resolved_survey, "uses_replicate_variance")
                    and resolved_survey.uses_replicate_variance
                ):
                    _survey_df = 0

                # Recompute survey_metadata from unit-level design so reported
                # effective_n/n_psu/df_survey match the inference actually run
                _unit_resolved = analytic.get("unit_resolved")
                if _unit_resolved is not None:
                    from diff_diff.survey import compute_survey_metadata

                    raw_w_unit = _unit_resolved.weights
                    survey_metadata = compute_survey_metadata(_unit_resolved, raw_w_unit)

                # Propagate replicate df override to survey_metadata for display
                # (but not the df=0 sentinel — keep metadata as None for undefined df)
                if _survey_df is not None and _survey_df != 0 and survey_metadata is not None:
                    if survey_metadata.df_survey != _survey_df:
                        survey_metadata.df_survey = _survey_df

                overall_att_t, overall_att_p, overall_att_ci = safe_inference(
                    overall_att, overall_att_se, self.alpha, df=_survey_df
                )
                overall_acrt_t, overall_acrt_p, overall_acrt_ci = safe_inference(
                    overall_acrt, overall_acrt_se, self.alpha, df=_survey_df
                )

                # Per-grid-point inference for dose-response
                for idx in range(n_grid):
                    _, _, ci = safe_inference(
                        agg_att_d[idx], att_d_se[idx], self.alpha, df=_survey_df
                    )
                    att_d_ci_lower[idx] = ci[0]
                    att_d_ci_upper[idx] = ci[1]

                    _, _, ci = safe_inference(
                        agg_acrt_d[idx], acrt_d_se[idx], self.alpha, df=_survey_df
                    )
                    acrt_d_ci_lower[idx] = ci[0]
                    acrt_d_ci_upper[idx] = ci[1]

                # Event study analytical SEs
                if event_study_effects is not None:
                    n_units = precomp["n_units"]
                    unit_sw = precomp.get("unit_survey_weights")

                    # Build unit-level ResolvedSurveyDesign once (reused per bin)
                    unit_resolved_es = None
                    if resolved_survey is not None:
                        row_idx = precomp["unit_first_panel_row"]
                        unit_resolved_es = resolved_survey.subset_to_units_by_row_idx(
                            row_idx, unit_weights=precomp.get("unit_survey_weights")
                        )

                    for e_val, info_e in event_study_effects.items():
                        # Collect (g,t) cells for this event-time bin
                        e_gts = [gt for gt in gt_results if gt[1] - gt[0] == e_val]
                        if not e_gts:
                            continue
                        # Weights within this bin: survey-weighted mass or n_treated
                        if unit_sw is not None:
                            unit_cohorts = precomp["unit_cohorts"]
                            ns = np.array(
                                [float(np.sum(unit_sw[unit_cohorts == gt[0]])) for gt in e_gts],
                                dtype=float,
                            )
                        else:
                            ns = np.array(
                                [gt_results[gt]["n_treated"] for gt in e_gts],
                                dtype=float,
                            )
                        total_n = ns.sum()
                        if total_n == 0:
                            continue
                        ws = ns / total_n

                        # Build per-unit IF for this event-time bin
                        if_es = np.zeros(n_units)
                        for idx_cell, gt in enumerate(e_gts):
                            b_info = gt_bootstrap_info.get(gt, {})
                            if not b_info:
                                continue
                            w = ws[idx_cell]
                            # Covariate path: the binarized event-study effect is
                            # att_glob, whose per-unit cell IF is precomputed.
                            cov_if = b_info.get("cov_if")
                            if cov_if is not None:
                                np.add.at(
                                    if_es,
                                    cov_if["cell_indices"],
                                    w * cov_if["if_att_glob"],
                                )
                                continue
                            treated_idx = b_info["treated_indices"]
                            control_idx = b_info["control_indices"]
                            n_t = b_info["n_treated"]
                            n_c = b_info["n_control"]
                            # Use survey-weighted masses when available
                            if "w_treated" in b_info:
                                n_t = b_info["w_treated"]
                                n_c = b_info["w_control"]
                            n_total_gt = n_t + n_c
                            p_1 = n_t / n_total_gt
                            p_0 = n_c / n_total_gt
                            att_glob_gt = b_info["att_glob"]
                            mu_0 = b_info["mu_0"]
                            delta_y_treated = b_info["delta_y_treated"]
                            ee_control = b_info["ee_control"]
                            sw_treated = b_info.get("w_treated_arr")

                            for k, uid in enumerate(treated_idx):
                                score_k = delta_y_treated[k] - att_glob_gt - mu_0
                                if sw_treated is not None:
                                    score_k = sw_treated[k] * score_k
                                if_es[uid] += w * score_k / p_1 / n_total_gt
                            for k, uid in enumerate(control_idx):
                                if_es[uid] -= w * ee_control[k] / p_0 / n_total_gt

                        # Compute SE: survey-aware TSL or standard sqrt(sum(IF^2))
                        if unit_resolved_es is not None:
                            if unit_resolved_es.uses_replicate_variance:
                                from diff_diff.survey import compute_replicate_if_variance

                                # Score-scale: psi = w * if_es (matches TSL bread)
                                psi_es = unit_resolved_es.weights * if_es
                                variance, _nv = compute_replicate_if_variance(
                                    psi_es, unit_resolved_es
                                )
                                es_se = (
                                    float(np.sqrt(max(variance, 0.0)))
                                    if np.isfinite(variance)
                                    else np.nan
                                )
                            else:
                                X_ones_es = np.ones((n_units, 1))
                                tsl_scale_es = float(unit_resolved_es.weights.sum())
                                if_es_tsl = if_es * tsl_scale_es
                                vcov_es = compute_survey_vcov(
                                    X_ones_es, if_es_tsl, unit_resolved_es
                                )
                                es_se = float(np.sqrt(np.abs(vcov_es[0, 0])))
                        else:
                            es_se = float(np.sqrt(np.sum(if_es**2)))

                        t_stat, p_val, ci_es = safe_inference(
                            info_e["effect"], es_se, self.alpha, df=_survey_df
                        )
                        info_e["se"] = es_se
                        info_e["t_stat"] = t_stat
                        info_e["p_value"] = p_val
                        info_e["conf_int"] = ci_es

        # 6. Assemble results
        dose_response_att = DoseResponseCurve(
            dose_grid=dvals,
            effects=agg_att_d,
            se=att_d_se,
            conf_int_lower=att_d_ci_lower,
            conf_int_upper=att_d_ci_upper,
            target="att",
            p_value=att_d_p,
            n_bootstrap=self.n_bootstrap,
            df_survey=_survey_df,
        )
        dose_response_acrt = DoseResponseCurve(
            dose_grid=dvals,
            effects=agg_acrt_d,
            se=acrt_d_se,
            conf_int_lower=acrt_d_ci_lower,
            conf_int_upper=acrt_d_ci_upper,
            target="acrt",
            p_value=acrt_d_p,
            n_bootstrap=self.n_bootstrap,
            df_survey=_survey_df,
        )

        # Strip bootstrap internals from gt_results
        clean_gt = {}
        for gt, r in gt_results.items():
            clean_gt[gt] = {k: v for k, v in r.items() if not k.startswith("_")}

        return ContinuousDiDResults(
            dose_response_att=dose_response_att,
            dose_response_acrt=dose_response_acrt,
            overall_att=overall_att,
            overall_att_se=overall_att_se,
            overall_att_t_stat=overall_att_t,
            overall_att_p_value=overall_att_p,
            overall_att_conf_int=overall_att_ci,
            overall_acrt=overall_acrt,
            overall_acrt_se=overall_acrt_se,
            overall_acrt_t_stat=overall_acrt_t,
            overall_acrt_p_value=overall_acrt_p,
            overall_acrt_conf_int=overall_acrt_ci,
            group_time_effects=clean_gt,
            dose_grid=dvals,
            groups=treatment_groups,
            time_periods=time_periods,
            n_obs=len(df),
            n_treated_units=int((unit_cohort > 0).sum()),
            n_control_units=n_control,
            alpha=self.alpha,
            control_group=self.control_group,
            covariates=self.covariates,
            estimation_method=self.estimation_method,
            pscore_trim=self.pscore_trim,
            epv_threshold=self.epv_threshold,
            pscore_fallback=self.pscore_fallback,
            degree=self.degree,
            num_knots=self.num_knots,
            base_period=self.base_period,
            anticipation=self.anticipation,
            n_bootstrap=self.n_bootstrap,
            bootstrap_weights=self.bootstrap_weights,
            seed=self.seed,
            rank_deficient_action=self.rank_deficient_action,
            event_study_effects=event_study_effects,
            survey_metadata=survey_metadata,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _precompute_structures(
        self,
        df: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        first_treat: str,
        dose: str,
        time_periods: List[Any],
        survey_weights: Optional[np.ndarray] = None,
        covariates: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Pivot to wide format and build lookup structures."""
        all_units = sorted(df[unit].unique())
        unit_to_idx = {u: i for i, u in enumerate(all_units)}
        n_units = len(all_units)
        n_periods = len(time_periods)
        period_to_col = {t: j for j, t in enumerate(time_periods)}

        # Outcome matrix: (n_units, n_periods)
        outcome_matrix = np.full((n_units, n_periods), np.nan)
        for _, row in df.iterrows():
            i = unit_to_idx[row[unit]]
            j = period_to_col[row[time]]
            outcome_matrix[i, j] = row[outcome]

        # Covariate cube: {period -> (n_units, n_cov)} read from the given
        # period (the cell reads its base period). Covariates may be
        # time-varying; the cell uses the base-period slice.
        covariate_by_period = None
        if covariates:
            cov_cube = np.full((n_units, n_periods, len(covariates)), np.nan)
            ui = df[unit].map(unit_to_idx).to_numpy()
            ti = df[time].map(period_to_col).to_numpy()
            cov_cube[ui, ti, :] = df[list(covariates)].to_numpy(dtype=float)
            covariate_by_period = {t: cov_cube[:, period_to_col[t], :] for t in time_periods}

        # Per-unit cohort and dose
        unit_cohorts = np.zeros(n_units, dtype=float)
        dose_vector = np.zeros(n_units, dtype=float)
        unit_first = df.groupby(unit).first()
        for u in all_units:
            i = unit_to_idx[u]
            unit_cohorts[i] = unit_first.loc[u, first_treat]
            dose_vector[i] = unit_first.loc[u, dose]

        # Build unit-to-first-panel-row mapping (for subsetting panel-level
        # arrays): the positional index of each unit's first row in df, aligned
        # to ``all_units`` (== ``unit_to_idx`` order since
        # ``unit_to_idx = {u: i for i, u in enumerate(all_units)}``).
        unit_first_panel_row = build_unit_first_row_index(df[unit].values, all_units)

        # Per-unit survey weights (take first obs per unit from panel data)
        unit_survey_weights = None
        if survey_weights is not None:
            unit_survey_weights = survey_weights[unit_first_panel_row]

        # Cohort masks
        cohort_masks = {}
        unique_cohorts = np.unique(unit_cohorts)
        for c in unique_cohorts:
            cohort_masks[c] = unit_cohorts == c

        never_treated_mask = unit_cohorts == 0

        return {
            "all_units": all_units,
            "unit_to_idx": unit_to_idx,
            "outcome_matrix": outcome_matrix,
            "period_to_col": period_to_col,
            "unit_cohorts": unit_cohorts,
            "dose_vector": dose_vector,
            "cohort_masks": cohort_masks,
            "never_treated_mask": never_treated_mask,
            "time_periods": time_periods,
            "n_units": n_units,
            "unit_survey_weights": unit_survey_weights,
            "unit_first_panel_row": unit_first_panel_row,
            "covariate_by_period": covariate_by_period,
        }

    # ------------------------------------------------------------------
    # Covariate adjustment (conditional parallel trends)
    # ------------------------------------------------------------------

    def _fit_covariate_adjustment(
        self,
        delta_y_treated: np.ndarray,
        delta_y_control: np.ndarray,
        X_treated_raw: np.ndarray,
        X_control_raw: np.ndarray,
        g: Any,
        t: Any,
    ) -> Optional[Dict[str, Any]]:
        """Covariate-adjusted control counterfactual for one (g,t) cell.

        Returns the per-treated counterfactual ``mu_0_vec`` and the nuisance
        pieces the influence function needs (reg: outcome-regression only; dr:
        + propensity and the DRDID doubly-robust per-unit IF). Missing/non-finite
        covariate values raise ``ValueError`` (fail-closed; ``fit()`` also rejects
        them up front — a per-cell fallback would mix conditional and
        unconditional estimands). ``reg`` and ``dr`` share the same OLS outcome
        regression on controls; ``dr`` adds a propensity model and a scalar
        augmentation ``eta_cont`` (a level term).
        """
        if np.any(np.isnan(X_treated_raw)) or np.any(np.isnan(X_control_raw)):
            # Defensive: fit() rejects non-finite covariates up front, so this
            # is an internal-invariant guard (no silent per-cell fallback).
            raise ValueError(
                f"Missing covariate values reached cell (g={g}, t={t}). "
                "ContinuousDiD requires complete covariates."
            )

        n_t = len(delta_y_treated)
        n_c = len(delta_y_control)
        Xt = np.column_stack([np.ones(n_t), X_treated_raw])
        Xc = np.column_stack([np.ones(n_c), X_control_raw])

        # Outcome regression on controls (OLS) — shared by reg and dr.
        gamma, _, _ = solve_ols(
            Xc,
            delta_y_control,
            return_vcov=False,
            rank_deficient_action=self.rank_deficient_action,
        )
        gamma = np.where(np.isnan(gamma), 0.0, gamma)
        mu_treated = Xt @ gamma
        or_resid_c = delta_y_control - Xc @ gamma  # per-control OR residual

        # OR-nuisance influence function: IF_gamma[k] = M_c^{-1} X_c[k] resid_c[k]
        Mc = (Xc.T @ Xc) / n_c
        Mc_inv, _, _ = _rank_guarded_inv(Mc)
        IF_gamma = (Xc @ Mc_inv) * or_resid_c[:, np.newaxis]  # (n_c, p), Mc_inv symmetric
        x_bar_treated = Xt.mean(axis=0)
        n_total = n_t + n_c

        eta_cont = 0.0
        dr_inf = None
        if self.estimation_method == "dr":
            D = np.concatenate([np.ones(n_t), np.zeros(n_c)])
            X_raw_all = np.vstack([X_treated_raw, X_control_raw])
            try:
                _, ps = solve_logit(
                    X_raw_all,
                    D,
                    rank_deficient_action=self.rank_deficient_action,
                    epv_threshold=self.epv_threshold,
                )
            except (np.linalg.LinAlgError, ValueError):
                # Fail closed by default; also honor an explicit
                # rank_deficient_action="error" request (mirrors CS).
                if self.pscore_fallback == "error" or self.rank_deficient_action == "error":
                    raise
                warnings.warn(
                    f"Propensity score estimation failed for (g={g}, t={t}); "
                    "falling back to an unconditional propensity for this cell "
                    "(this cell is now reg-like, not doubly-robust). "
                    "Consider estimation_method='reg' to avoid propensity scores.",
                    UserWarning,
                    stacklevel=3,
                )
                ps = np.full(n_total, n_t / n_total)
            ps = np.clip(ps, self.pscore_trim, 1 - self.pscore_trim)
            odds_c = ps[n_t:] / (1 - ps[n_t:])
            eta_cont = float(np.sum(odds_c * or_resid_c) / np.sum(odds_c))
            dY_all = np.concatenate([delta_y_treated, delta_y_control])
            dr_inf = self._dr_cell_inf_func(dY_all, D, np.vstack([Xt, Xc]), gamma, ps)

        return {
            "mu_0_vec": mu_treated + eta_cont,  # per-treated counterfactual
            "Xt": Xt,  # (n_t, p) treated covariates incl. intercept
            "IF_gamma": IF_gamma,  # (n_c, p)
            "x_bar_treated": x_bar_treated,  # (p,)
            "eta_cont": eta_cont,
            "dr_inf": dr_inf,  # (n_total,) DRDID att.inf.func, treated-then-control
            "n_total": n_total,
        }

    def _dr_cell_inf_func(
        self,
        dY: np.ndarray,
        D: np.ndarray,
        X: np.ndarray,
        gamma: np.ndarray,
        ps: np.ndarray,
    ) -> np.ndarray:
        """DRDID ``drdid_panel`` doubly-robust per-unit influence function.

        Direct port of ``DRDID::drdid_panel$att.inf.func`` (unit weights = 1,
        propensity already clipped so no drop-trimming). Units are ordered
        treated-then-control. Validated to ~1e-13 against DRDID in the spike.
        """
        n = len(D)
        out_delta = X @ gamma
        w_treat = D
        w_cont = ps * (1 - D) / (1 - ps)
        dr_treat = w_treat * (dY - out_delta)
        dr_cont = w_cont * (dY - out_delta)
        eta_treat = dr_treat.mean() / w_treat.mean()
        eta_cont = dr_cont.mean() / w_cont.mean()
        weights_ols = 1.0 - D
        wols_eX = (weights_ols * (dY - out_delta))[:, np.newaxis] * X
        XpX = ((weights_ols[:, np.newaxis] * X).T @ X) / n
        XpX_inv, _, _ = _rank_guarded_inv(XpX)
        asy_wols = wols_eX @ XpX_inv
        W = ps * (1 - ps)
        score_ps = (D - ps)[:, np.newaxis] * X
        Hess, _, _ = _rank_guarded_inv((X.T @ (W[:, np.newaxis] * X)) / n)
        asy_ps = score_ps @ Hess
        inf_treat_1 = dr_treat - w_treat * eta_treat
        M1 = (w_treat[:, np.newaxis] * X).mean(axis=0)
        inf_treat = (inf_treat_1 - asy_wols @ M1) / w_treat.mean()
        inf_cont_1 = dr_cont - w_cont * eta_cont
        M2 = (w_cont[:, np.newaxis] * (dY - out_delta - eta_cont)[:, np.newaxis] * X).mean(axis=0)
        M3 = (w_cont[:, np.newaxis] * X).mean(axis=0)
        inf_control = (inf_cont_1 + asy_ps @ M2 - asy_wols @ M3) / w_cont.mean()
        return inf_treat - inf_control

    def _covariate_cell_influence(
        self,
        cov_adj: Dict[str, Any],
        delta_tilde_y: np.ndarray,
        att_glob: float,
        residuals: np.ndarray,
        Psi: np.ndarray,
        bread: np.ndarray,
        Psi_eval: np.ndarray,
        dPsi_eval: np.ndarray,
        dpsi_bar: np.ndarray,
        treated_indices: np.ndarray,
        control_indices: np.ndarray,
        n_t: int,
        n_c: int,
    ) -> Dict[str, Any]:
        """Per-cell-unit influence functions for the covariate-adjusted estimands.

        Returns arrays aligned to ``cell_indices = [treated_indices,
        control_indices]`` in the 1/n convention (the aggregator scatters them
        with the cell weight and the SE is ``sqrt(sum(IF^2))``). At p=1 (intercept
        only) this reduces exactly to the unconditional control-IF path.
        """
        Xt = cov_adj["Xt"]  # (n_t, p)
        IF_gamma = cov_adj["IF_gamma"]  # (n_c, p)
        x_bar_treated = cov_adj["x_bar_treated"]  # (p,)

        # Treated: beta perturbation from the B-spline residual score.
        beta_if_t = (Psi * residuals[:, np.newaxis]) @ bread / n_t  # (n_t, K), bread symmetric
        att_d_if_t = beta_if_t @ Psi_eval.T  # (n_t, n_grid)
        acrt_d_if_t = beta_if_t @ dPsi_eval.T
        acrt_glob_if_t = beta_if_t @ dpsi_bar  # (n_t,)
        att_glob_if_t = (delta_tilde_y - att_glob) / n_t  # (n_t,)

        # Control: enters through the outcome-regression nuisance gamma_hat.
        # E_T[Psi X'] (K x p) replaces the scalar psi_bar; E_T[X'] the scalar 1.
        Psi_X_bar = (Psi.T @ Xt) / n_t  # (K, p)
        beta_if_c = -((IF_gamma @ Psi_X_bar.T) @ bread) / n_c  # (n_c, K)
        att_d_if_c = beta_if_c @ Psi_eval.T
        acrt_d_if_c = beta_if_c @ dPsi_eval.T
        acrt_glob_if_c = beta_if_c @ dpsi_bar
        att_glob_if_c = -(IF_gamma @ x_bar_treated) / n_c  # (n_c,)

        if_att_glob = np.concatenate([att_glob_if_t, att_glob_if_c])
        if_acrt_glob = np.concatenate([acrt_glob_if_t, acrt_glob_if_c])
        if_att_d = np.vstack([att_d_if_t, att_d_if_c])  # (n_total, n_grid)
        if_acrt_d = np.vstack([acrt_d_if_t, acrt_d_if_c])

        if self.estimation_method == "dr":
            # dr shares the reg curve shape (ACRT identical); att_glob / ATT(d)
            # differ by the augmentation eta_cont. Ground att_glob's IF in the
            # validated DRDID doubly-robust IF and shift ATT(d) uniformly by the
            # augmentation IF (= reg att_glob IF - dr att_glob IF). ACRT untouched.
            n_total = cov_adj["n_total"]
            dr_att_glob_if = cov_adj["dr_inf"] / n_total  # (n_total,)
            if_eta = if_att_glob - dr_att_glob_if  # augmentation IF, per unit
            if_att_d = if_att_d - if_eta[:, np.newaxis]
            if_att_glob = dr_att_glob_if

        return {
            "cell_indices": np.concatenate([treated_indices, control_indices]),
            "if_att_glob": if_att_glob,
            "if_acrt_glob": if_acrt_glob,
            "if_att_d": if_att_d,
            "if_acrt_d": if_acrt_d,
        }

    def _compute_dose_response_gt(
        self,
        precomp: Dict[str, Any],
        g: Any,
        t: Any,
        knots: np.ndarray,
        degree: int,
        dvals: np.ndarray,
        survey_weights: Optional[np.ndarray] = None,
        resolved_survey: object = None,
    ) -> Optional[Dict[str, Any]]:
        """Compute dose-response for a single (g,t) cell."""
        period_to_col = precomp["period_to_col"]
        outcome_matrix = precomp["outcome_matrix"]
        unit_cohorts = precomp["unit_cohorts"]
        dose_vector = precomp["dose_vector"]
        never_treated_mask = precomp["never_treated_mask"]
        time_periods = precomp["time_periods"]

        # Base period selection
        is_post = t >= g - self.anticipation
        if self.base_period == "varying":
            if is_post:
                base_t = g - 1 - self.anticipation
            else:
                # Pre-treatment: use t-1
                t_idx = time_periods.index(t)
                if t_idx == 0:
                    return None  # No prior period
                base_t = time_periods[t_idx - 1]
        else:
            # Universal base period
            base_t = g - 1 - self.anticipation

        if base_t not in period_to_col or t not in period_to_col:
            return None

        col_t = period_to_col[t]
        col_base = period_to_col[base_t]

        # Treated units: first_treat == g and dose > 0
        treated_mask = (unit_cohorts == g) & (dose_vector > 0)
        n_treated = int(np.sum(treated_mask))
        if n_treated == 0:
            return None

        # Control units
        if self.control_group == "never_treated":
            control_mask = never_treated_mask
        else:
            # Not-yet-treated: never-treated + first_treat > t
            control_mask = never_treated_mask | (
                (unit_cohorts > t + self.anticipation) & (unit_cohorts != g)
            )
        n_control = int(np.sum(control_mask))
        if n_control == 0:
            warnings.warn(
                f"No control units for (g={g}, t={t}). Skipping.",
                UserWarning,
                stacklevel=3,
            )
            return None

        # Outcome changes
        delta_y_treated = (
            outcome_matrix[treated_mask, col_t] - outcome_matrix[treated_mask, col_base]
        )
        delta_y_control = (
            outcome_matrix[control_mask, col_t] - outcome_matrix[control_mask, col_base]
        )

        # Subset survey weights to the cell
        w_treated = None
        w_control = None
        if survey_weights is not None:
            w_treated = survey_weights[treated_mask]
            w_control = survey_weights[control_mask]
            # Guard against zero effective mass (e.g., after subpopulation)
            if np.sum(w_treated) <= 0 or np.sum(w_control) <= 0:
                return {
                    "att_glob": np.nan,
                    "acrt_glob": np.nan,
                    "n_treated": 0,
                    "n_control": 0,
                    "att_d": np.full(len(dvals), np.nan),
                    "acrt_d": np.full(len(dvals), np.nan),
                }

        # Control counterfactual.
        #   - No covariates: scalar control mean mu_0 (unconditional PT).
        #   - Covariates (reg/dr): per-treated covariate-adjusted counterfactual
        #     mu_0_vec = X_i'gamma_hat (+ scalar DR augmentation eta_cont for dr),
        #     under conditional PT. `cov_adj` carries the nuisance pieces the
        #     influence function needs. Survey weights are rejected upstream on
        #     the covariate path, so w_control/w_treated are None here.
        covariate_by_period = precomp.get("covariate_by_period")
        cov_adj = None
        if covariate_by_period is not None:
            cov_adj = self._fit_covariate_adjustment(
                delta_y_treated,
                delta_y_control,
                covariate_by_period[base_t][treated_mask],
                covariate_by_period[base_t][control_mask],
                g,
                t,
            )

        if cov_adj is not None:
            mu_0 = float(np.mean(delta_y_control))  # retained for metadata only
            delta_tilde_y = delta_y_treated - cov_adj["mu_0_vec"]
        elif w_control is not None:
            mu_0 = float(np.average(delta_y_control, weights=w_control))
            delta_tilde_y = delta_y_treated - mu_0
        else:
            mu_0 = float(np.mean(delta_y_control))
            delta_tilde_y = delta_y_treated - mu_0

        # Treated doses
        treated_doses = dose_vector[treated_mask]

        # B-spline OLS
        Psi = bspline_design_matrix(treated_doses, knots, degree, include_intercept=True)
        n_basis = Psi.shape[1]

        # Check for all-same dose
        if np.all(treated_doses == treated_doses[0]):
            warnings.warn(
                f"All treated doses identical in (g={g}, t={t}). " "ACRT(d) will be 0 everywhere.",
                UserWarning,
                stacklevel=3,
            )

        # Skip if not enough treated units for OLS (need n > K for residual df)
        # When survey weights are present, use positive-weight count as
        # the effective sample size — subpopulation() can zero weights
        # leaving rows present but the weighted regression underidentified.
        n_eff = int(np.count_nonzero(w_treated > 0)) if w_treated is not None else n_treated
        if n_eff <= n_basis:
            label = "positive-weight treated units" if w_treated is not None else "treated units"
            warnings.warn(
                f"Not enough {label} ({n_eff}) for {n_basis} basis functions "
                f"in (g={g}, t={t}). Skipping cell.",
                UserWarning,
                stacklevel=3,
            )
            return None

        # OLS or WLS regression
        if w_treated is not None:
            # WLS: apply sqrt(w) transformation
            sqrt_w = np.sqrt(w_treated)
            Psi_w = Psi * sqrt_w[:, np.newaxis]
            delta_tilde_y_w = delta_tilde_y * sqrt_w
            beta_hat, _, _ = solve_ols(
                Psi_w,
                delta_tilde_y_w,
                return_vcov=False,
                rank_deficient_action=self.rank_deficient_action,
            )
            # Residuals on original scale (for influence functions)
            beta_pred_tmp = np.where(np.isnan(beta_hat), 0.0, beta_hat)
            residuals = delta_tilde_y - Psi @ beta_pred_tmp
        else:
            beta_hat, residuals, _ = solve_ols(
                Psi,
                delta_tilde_y,
                return_vcov=False,
                rank_deficient_action=self.rank_deficient_action,
            )

        # For prediction: zero out NaN (dropped rank-deficient columns).
        # solve_ols sets dropped-column coefficients to NaN (R convention);
        # zeroing them produces correct predictions: ATT(d) = intercept
        # (constant), ACRT(d) = 0 (derivative of intercept is 0).
        beta_pred = np.where(np.isnan(beta_hat), 0.0, beta_hat)

        # Evaluate ATT(d) and ACRT(d) at dvals
        Psi_eval = bspline_design_matrix(dvals, knots, degree, include_intercept=True)
        dPsi_eval = bspline_derivative_design_matrix(dvals, knots, degree, include_intercept=True)

        att_d = Psi_eval @ beta_pred
        acrt_d = dPsi_eval @ beta_pred

        # Summary parameters. With covariates, att_glob = mean_T(delta_tilde_y)
        # (reg: mean_T(dY - X'gamma); dr: additionally minus the augmentation
        # eta_cont, already folded into delta_tilde_y via mu_0_vec).
        if cov_adj is not None:
            att_glob = float(np.mean(delta_tilde_y))
        elif w_treated is not None:
            att_glob = float(np.average(delta_y_treated, weights=w_treated) - mu_0)
        else:
            att_glob = float(np.mean(delta_y_treated) - mu_0)

        # ACRT^{glob}: plug-in average of ACRT(D_i) for treated
        dPsi_treated = bspline_derivative_design_matrix(
            treated_doses, knots, degree, include_intercept=True
        )
        if w_treated is not None:
            acrt_glob = float(np.average(dPsi_treated @ beta_pred, weights=w_treated))
        else:
            acrt_glob = float(np.mean(dPsi_treated @ beta_pred))

        # Store bootstrap info for influence function computation
        # bread = (Psi'WPsi / n_treated)^{-1} when survey, (Psi'Psi / n_treated)^{-1} otherwise
        # Bread = (Psi'WPsi / mass)^{-1} via the shared rank-guarded inverse:
        # np.linalg.inv only raises on an *exactly* singular Gram, so a *near*-
        # singular B-spline design (clustered doses / near-duplicate knots)
        # previously returned a garbage inverse (~1e13) -> garbage SE. The prior
        # `pinv` fallback was both minimum-norm (not the column-drop / near-
        # collinear limit) and *silent*. `_rank_guarded_inv` truncates redundant
        # directions on the equilibrated Gram -> finite SE on the identified
        # subspace (NaN only at rank 0), matching the covariate IF rank-guard.
        if w_treated is not None:
            w_treated_sum = float(np.sum(w_treated))
            PtWP = Psi.T @ (Psi * w_treated[:, np.newaxis])
            # Normalize bread by weighted mass (not raw count) for consistency
            # with downstream IF score denominators that also use weighted mass
            bread, n_dropped, _ = _rank_guarded_inv(PtWP / w_treated_sum)
        else:
            PtP = Psi.T @ Psi
            bread, n_dropped, _ = _rank_guarded_inv(PtP / n_treated)
        if n_dropped:
            warnings.warn(
                "ContinuousDiD ACRT variance: the B-spline design Gram is "
                f"rank-deficient ({n_dropped} redundant direction(s) dropped); "
                "rank-reducing to a finite SE on the identified subspace. "
                "Analytical SEs reflect the reduced rank (NaN if rank 0).",
                UserWarning,
                stacklevel=2,
            )

        # ee_treated: per-unit estimating equation vectors (K-vector per unit)
        # For WLS (survey weights), the score is w_i * X_i * u_i to match the
        # weighted bread inv(X'WX / sum(w)).  Without this factor the sandwich
        # is inconsistent.  For OLS (no survey weights), the score is X_i * u_i.
        if w_treated is not None:
            ee_treated = Psi * (w_treated * residuals)[:, np.newaxis]  # (n_treated, K)
        else:
            ee_treated = Psi * residuals[:, np.newaxis]  # (n_treated, K)

        # ee_control: per-unit deviation from control mean (weighted for WLS)
        if w_control is not None:
            ee_control = w_control * (delta_y_control - mu_0)  # (n_control,)
        else:
            ee_control = delta_y_control - mu_0  # (n_control,)

        # psi_bar: mean basis vector for treated (weighted when survey)
        if w_treated is not None:
            psi_bar = np.average(Psi, axis=0, weights=w_treated)
        else:
            psi_bar = np.mean(Psi, axis=0)  # (K,)

        # Unit indices for bootstrap
        treated_indices = np.where(treated_mask)[0]
        control_indices = np.where(control_mask)[0]

        # dpsi_bar: mean derivative basis vector (weighted when survey)
        if w_treated is not None:
            dpsi_bar = np.average(dPsi_treated, axis=0, weights=w_treated)
        else:
            dpsi_bar = np.mean(dPsi_treated, axis=0)

        # Covariate influence-function arrays (per cell unit, treated-then-control,
        # in the 1/n "sum-of-squares SE" convention). Reg uses the p-vector OR
        # generalization of the unconditional control IF (which is its p=1 case);
        # dr reuses the reg curve shape and grounds att_glob + the augmentation IF
        # in the validated DRDID doubly-robust per-unit IF. See REGISTRY.
        cov_if = None
        if cov_adj is not None:
            cov_if = self._covariate_cell_influence(
                cov_adj,
                delta_tilde_y,
                att_glob,
                residuals,
                Psi,
                bread,
                Psi_eval,
                dPsi_eval,
                dpsi_bar,
                treated_indices,
                control_indices,
                n_treated,
                n_control,
            )

        bootstrap_info = {
            "bread": bread,
            "ee_treated": ee_treated,
            "ee_control": ee_control,
            "psi_bar": psi_bar,
            "dpsi_bar": dpsi_bar,
            "beta_hat": beta_hat,
            "beta_pred": beta_pred,
            "treated_indices": treated_indices,
            "control_indices": control_indices,
            "n_treated": n_treated,
            "n_control": n_control,
            "Psi_eval": Psi_eval,
            "dPsi_eval": dPsi_eval,
            "dPsi_treated": dPsi_treated,
            "delta_y_treated": delta_y_treated,
            "delta_y_control": delta_y_control,
            "mu_0": mu_0,
            "att_glob": att_glob,
            "acrt_glob": acrt_glob,
            "cov_if": cov_if,
        }

        # Store survey-weighted masses and per-unit arrays for IF linearization
        if w_treated is not None:
            bootstrap_info["w_treated"] = float(np.sum(w_treated))
            bootstrap_info["w_control"] = float(np.sum(w_control))
            bootstrap_info["w_treated_arr"] = w_treated
            bootstrap_info["w_control_arr"] = w_control

        return {
            "att_d": att_d,
            "acrt_d": acrt_d,
            "att_glob": att_glob,
            "acrt_glob": acrt_glob,
            "beta_hat": beta_hat,
            "n_treated": n_treated,
            "n_control": n_control,
            "_bootstrap_info": bootstrap_info,
        }

    def _aggregate_event_study(
        self,
        gt_results: Dict[Tuple, Dict],
        gt_bootstrap_info: Dict[Tuple, Dict] = None,
        unit_survey_weights: Optional[np.ndarray] = None,
        unit_cohorts: Optional[np.ndarray] = None,
        anticipation: int = 0,
    ) -> Dict[int, Dict[str, Any]]:
        """Aggregate binarized ATT_glob by relative period."""
        effects_by_e: Dict[int, List[Tuple[float, float, Tuple]]] = {}

        for (g, t), r in gt_results.items():
            e = t - g
            if anticipation > 0 and e < -anticipation:
                continue
            if e not in effects_by_e:
                effects_by_e[e] = []
            # Compute weight for this (g,t) cell
            if unit_survey_weights is not None and unit_cohorts is not None:
                # Survey-weighted: sum of survey weights for treated units in group g
                g_mask = unit_cohorts == g
                cell_weight = float(np.sum(unit_survey_weights[g_mask]))
            else:
                cell_weight = float(r["n_treated"])
            effects_by_e[e].append((r["att_glob"], cell_weight, (g, t)))

        result = {}
        for e, entries in sorted(effects_by_e.items()):
            effects = np.array([x[0] for x in entries])
            weights = np.array([x[1] for x in entries])
            if np.sum(weights) > 0:
                w = weights / np.sum(weights)
                agg = float(np.sum(w * effects))
            else:
                agg = np.nan
            result[e] = {
                "effect": agg,
                "se": np.nan,
                "t_stat": np.nan,
                "p_value": np.nan,
                "conf_int": (np.nan, np.nan),
            }
        return result

    def _compute_analytical_se(
        self,
        precomp: Dict[str, Any],
        gt_results: Dict[Tuple, Dict],
        gt_bootstrap_info: Dict[Tuple, Dict],
        post_gt: Dict[Tuple, Dict],
        cell_weights: Dict[Tuple, float],
        knots: np.ndarray,
        degree: int,
        dvals: np.ndarray,
        agg_att_d: np.ndarray,
        agg_acrt_d: np.ndarray,
        resolved_survey: object = None,
    ) -> Dict[str, Any]:
        """Compute analytical SEs using influence functions."""
        n_units = precomp["n_units"]
        n_grid = len(dvals)

        # Build per-unit influence functions for aggregated parameters
        # IF_i for overall ATT_glob (binarized)
        if_att_glob = np.zeros(n_units)
        if_acrt_glob = np.zeros(n_units)
        if_att_d = np.zeros((n_units, n_grid))
        if_acrt_d = np.zeros((n_units, n_grid))

        for gt, w in cell_weights.items():
            if w == 0:
                continue
            info = gt_bootstrap_info[gt]
            if not info:
                continue
            # Covariate path: scatter the pre-computed per-unit cell IFs
            # (unconditional / survey path is untouched below).
            cov_if = info.get("cov_if")
            if cov_if is not None:
                idx = cov_if["cell_indices"]
                np.add.at(if_att_glob, idx, w * cov_if["if_att_glob"])
                np.add.at(if_acrt_glob, idx, w * cov_if["if_acrt_glob"])
                np.add.at(if_att_d, idx, w * cov_if["if_att_d"])
                np.add.at(if_acrt_d, idx, w * cov_if["if_acrt_d"])
                continue
            treated_idx = info["treated_indices"]
            control_idx = info["control_indices"]
            n_t = info["n_treated"]
            n_c = info["n_control"]
            # Use survey-weighted masses when available
            if "w_treated" in info:
                n_t = info["w_treated"]
                n_c = info["w_control"]
            bread = info["bread"]
            ee_treated = info["ee_treated"]
            ee_control = info["ee_control"]
            psi_bar = info["psi_bar"]
            dpsi_bar = info["dpsi_bar"]
            Psi_eval = info["Psi_eval"]
            dPsi_eval = info["dPsi_eval"]
            att_glob_gt = info["att_glob"]
            mu_0 = info["mu_0"]
            delta_y_treated = info["delta_y_treated"]
            # Per-unit survey weight array (None when no survey)
            sw_treated = info.get("w_treated_arr")

            n_total = n_t + n_c
            p_1 = n_t / n_total
            p_0 = n_c / n_total

            # IF for ATT_glob (binarized DiD)
            # When survey weights are present, each unit's score includes its
            # survey weight w_k so the sandwich is consistent with the weighted
            # estimand.  ee_control already contains the w_k factor (set in
            # _compute_dose_response_gt); delta_y_treated needs it here.
            for k, idx in enumerate(treated_idx):
                score_k = delta_y_treated[k] - att_glob_gt - mu_0
                if sw_treated is not None:
                    score_k = sw_treated[k] * score_k
                if_att_glob[idx] += w * score_k / p_1 / n_total
            for k, idx in enumerate(control_idx):
                if_att_glob[idx] -= w * ee_control[k] / p_0 / n_total

            # IF for beta perturbation → ATT(d) and ACRT(d)
            # beta perturbation from treated: bread @ (1/n_t) * sum w_i * ee_treated_i
            # beta perturbation from control: -bread @ psi_bar * (1/n_c) * sum w_i * ee_control_i
            # ATT_b(d) = Psi_eval @ beta_b  => IF_i(d) contribution

            # Treated unit contributions to beta
            for k, idx in enumerate(treated_idx):
                beta_pert = bread @ ee_treated[k] / n_t
                if_att_d[idx] += w * (Psi_eval @ beta_pert)
                if_acrt_d[idx] += w * (dPsi_eval @ beta_pert)

            # Control unit contributions to beta (through mu_0)
            for k, idx in enumerate(control_idx):
                beta_pert = -bread @ psi_bar * ee_control[k] / n_c
                if_att_d[idx] += w * (Psi_eval @ beta_pert)
                if_acrt_d[idx] += w * (dPsi_eval @ beta_pert)

            # ACRT_glob IF: (1/n_t) sum_j dpsi(D_j)' @ beta_pert
            for k, idx in enumerate(treated_idx):
                beta_pert = bread @ ee_treated[k] / n_t
                if_acrt_glob[idx] += w * float(dpsi_bar @ beta_pert)
            for k, idx in enumerate(control_idx):
                beta_pert = -bread @ psi_bar * ee_control[k] / n_c
                if_acrt_glob[idx] += w * float(dpsi_bar @ beta_pert)

        # Compute SEs from influence functions
        if resolved_survey is not None:
            # Survey design: use TSL variance on the aggregate influence functions.
            # The IFs serve as "residuals" in the TSL sandwich; X is a column of ones
            # (the estimand is a scalar/vector mean of the IFs).
            #
            # The resolved_survey has panel-level arrays (n_obs = n_units * n_periods),
            # but influence functions are unit-level (n_units). Build a unit-level
            # ResolvedSurveyDesign by subsetting to one obs per unit.
            row_idx = precomp["unit_first_panel_row"]
            unit_resolved = resolved_survey.subset_to_units_by_row_idx(
                row_idx, unit_weights=precomp.get("unit_survey_weights")
            )

            X_ones = np.ones((n_units, 1))

            if unit_resolved.uses_replicate_variance:
                # Replicate-weight variance: score-scale IFs to match TSL bread.
                # TSL path does: scores = w * (if * tsl_scale), bread = 1/sum(w)^2
                # Equivalent psi for replicate: w * if_vals * tsl_scale / sum(w) = w * if_vals
                from diff_diff.survey import compute_replicate_if_variance

                _w_rep = unit_resolved.weights
                _rep_n_valid = unit_resolved.n_replicates  # track effective count

                def _rep_se(if_vals):
                    nonlocal _rep_n_valid
                    psi_scaled = _w_rep * if_vals
                    v, nv = compute_replicate_if_variance(psi_scaled, unit_resolved)
                    _rep_n_valid = min(_rep_n_valid, nv)  # worst-case valid count
                    return float(np.sqrt(max(v, 0.0))) if np.isfinite(v) else np.nan

                overall_att_se = _rep_se(if_att_glob)
                overall_acrt_se = _rep_se(if_acrt_glob)
                att_d_se = np.zeros(n_grid)
                acrt_d_se = np.zeros(n_grid)
                for d_idx in range(n_grid):
                    att_d_se[d_idx] = _rep_se(if_att_d[:, d_idx])
                    acrt_d_se[d_idx] = _rep_se(if_acrt_d[:, d_idx])
            else:
                # TSL: rescale IFs from 1/n convention to score scale for sandwich.
                tsl_scale = float(unit_resolved.weights.sum())
                if_att_glob_tsl = if_att_glob * tsl_scale
                if_acrt_glob_tsl = if_acrt_glob * tsl_scale
                if_att_d_tsl = if_att_d * tsl_scale
                if_acrt_d_tsl = if_acrt_d * tsl_scale

                vcov_att = compute_survey_vcov(X_ones, if_att_glob_tsl, unit_resolved)
                overall_att_se = float(np.sqrt(np.abs(vcov_att[0, 0])))

                vcov_acrt = compute_survey_vcov(X_ones, if_acrt_glob_tsl, unit_resolved)
                overall_acrt_se = float(np.sqrt(np.abs(vcov_acrt[0, 0])))

                att_d_se = np.zeros(n_grid)
                acrt_d_se = np.zeros(n_grid)
                for d_idx in range(n_grid):
                    vcov_d = compute_survey_vcov(X_ones, if_att_d_tsl[:, d_idx], unit_resolved)
                    att_d_se[d_idx] = float(np.sqrt(np.abs(vcov_d[0, 0])))

                    vcov_d = compute_survey_vcov(X_ones, if_acrt_d_tsl[:, d_idx], unit_resolved)
                    acrt_d_se[d_idx] = float(np.sqrt(np.abs(vcov_d[0, 0])))
        else:
            # SE = sqrt(sum(IF_i^2)), matching CallawaySantAnna's convention
            # (per-unit IFs already contain 1/n_t, 1/n_c scaling)
            overall_att_se = float(np.sqrt(np.sum(if_att_glob**2)))
            overall_acrt_se = float(np.sqrt(np.sum(if_acrt_glob**2)))

            att_d_se = np.sqrt(np.sum(if_att_d**2, axis=0))
            acrt_d_se = np.sqrt(np.sum(if_acrt_d**2, axis=0))

        # Return unit-level survey df and resolved design for metadata recomputation
        # Only override with n_valid-based df when replicates were actually dropped
        if (
            resolved_survey is not None
            and hasattr(resolved_survey, "uses_replicate_variance")
            and resolved_survey.uses_replicate_variance
        ):
            if _rep_n_valid < unit_resolved.n_replicates:
                unit_df_survey = _rep_n_valid - 1 if _rep_n_valid > 1 else None
            else:
                unit_df_survey = unit_resolved.df_survey
        else:
            unit_df_survey = unit_resolved.df_survey if resolved_survey is not None else None

        return {
            "overall_att_se": overall_att_se,
            "overall_acrt_se": overall_acrt_se,
            "att_d_se": att_d_se,
            "acrt_d_se": acrt_d_se,
            "df_survey": unit_df_survey,
            "unit_resolved": unit_resolved if resolved_survey is not None else None,
        }

    def _run_bootstrap(
        self,
        precomp: Dict[str, Any],
        gt_results: Dict[Tuple, Dict],
        gt_bootstrap_info: Dict[Tuple, Dict],
        post_gt: Dict[Tuple, Dict],
        cell_weights: Dict[Tuple, float],
        knots: np.ndarray,
        degree: int,
        dvals: np.ndarray,
        original_att: float,
        original_acrt: float,
        original_att_d: np.ndarray,
        original_acrt_d: np.ndarray,
        event_study_effects: Optional[Dict[int, Dict]],
        resolved_survey: object = None,
    ) -> Dict[str, Any]:
        """Run multiplier bootstrap inference."""
        if self.n_bootstrap < 50:
            warnings.warn(
                f"n_bootstrap={self.n_bootstrap} is low. Consider n_bootstrap >= 199 "
                "for reliable inference.",
                UserWarning,
                stacklevel=3,
            )

        # Reject replicate-weight designs for bootstrap — replicate variance
        # is an analytical alternative to bootstrap, not compatible with it
        if (
            resolved_survey is not None
            and hasattr(resolved_survey, "uses_replicate_variance")
            and resolved_survey.uses_replicate_variance
        ):
            raise NotImplementedError(
                "ContinuousDiD bootstrap (n_bootstrap > 0) is not supported "
                "with replicate-weight survey designs. Replicate weights provide "
                "analytical variance; use n_bootstrap=0 instead."
            )

        rng = np.random.default_rng(self.seed)
        n_units = precomp["n_units"]
        n_grid = len(dvals)

        # Build unit-level ResolvedSurveyDesign for survey-aware bootstrap
        unit_resolved = None
        if resolved_survey is not None:
            row_idx = precomp["unit_first_panel_row"]
            unit_resolved = resolved_survey.subset_to_units_by_row_idx(
                row_idx, unit_weights=precomp.get("unit_survey_weights")
            )

        # Generate bootstrap weights — PSU-level when survey design is present
        _use_survey_bootstrap = unit_resolved is not None and (
            unit_resolved.strata is not None
            or unit_resolved.psu is not None
            or unit_resolved.fpc is not None
        )

        if _use_survey_bootstrap:
            from diff_diff.bootstrap_utils import (
                generate_survey_multiplier_weights_batch,
            )

            psu_weights, psu_ids = generate_survey_multiplier_weights_batch(
                self.n_bootstrap, unit_resolved, self.bootstrap_weights, rng
            )
            # Build unit -> PSU column map
            if unit_resolved.psu is not None:
                psu_id_to_col = {int(p): c for c, p in enumerate(psu_ids)}
                unit_to_psu_col = np.array(
                    [psu_id_to_col[int(unit_resolved.psu[i])] for i in range(n_units)]
                )
            else:
                unit_to_psu_col = np.arange(n_units)
            all_weights = psu_weights[:, unit_to_psu_col]
        else:
            all_weights = generate_bootstrap_weights_batch(
                self.n_bootstrap, n_units, self.bootstrap_weights, rng
            )

        boot_att_glob = np.zeros(self.n_bootstrap)
        boot_acrt_glob = np.zeros(self.n_bootstrap)
        boot_att_d = np.zeros((self.n_bootstrap, n_grid))
        boot_acrt_d = np.zeros((self.n_bootstrap, n_grid))

        # Event study bootstrap — compute weights per event-time bin
        es_keys = sorted(event_study_effects.keys()) if event_study_effects else []
        boot_es = {e: np.zeros(self.n_bootstrap) for e in es_keys}
        # Per-(g,t) weight within event-time bin — use survey-weighted cohort
        # masses when available, matching _aggregate_event_study.
        unit_sw = precomp.get("unit_survey_weights")
        unit_cohorts = precomp["unit_cohorts"]
        es_cell_weights: Dict[Tuple, float] = {}
        if event_study_effects is not None:
            from collections import defaultdict

            es_bin_total: Dict[int, float] = defaultdict(float)
            for gt, r in gt_results.items():
                g_val, t_val = gt
                e = t_val - g_val
                if self.anticipation > 0 and e < -self.anticipation:
                    continue
                if unit_sw is not None:
                    g_mask = unit_cohorts == g_val
                    cell_mass = float(np.sum(unit_sw[g_mask]))
                else:
                    cell_mass = float(r["n_treated"])
                es_bin_total[e] += cell_mass
            for gt, r in gt_results.items():
                g_val, t_val = gt
                e = t_val - g_val
                if self.anticipation > 0 and e < -self.anticipation:
                    continue
                if unit_sw is not None:
                    g_mask = unit_cohorts == g_val
                    cell_mass = float(np.sum(unit_sw[g_mask]))
                else:
                    cell_mass = float(r["n_treated"])
                if es_bin_total[e] > 0:
                    es_cell_weights[gt] = cell_mass / es_bin_total[e]

        # Helper to bootstrap a single (g,t) cell
        def _bootstrap_gt_cell(gt, info):
            """Returns att_glob_b array (B,) for this cell."""
            # Covariate path: the multiplier bootstrap perturbs the same per-unit
            # cell influence functions the analytical SE uses.
            cov_if = info.get("cov_if")
            if cov_if is not None:
                xi = all_weights[:, cov_if["cell_indices"]]  # (B, n_cell)
                beta_pred_c = info["beta_pred"]
                cell_att_d = info["Psi_eval"] @ beta_pred_c
                cell_acrt_d = info["dPsi_eval"] @ beta_pred_c
                att_d_b = cell_att_d[np.newaxis, :] + xi @ cov_if["if_att_d"]
                acrt_d_b = cell_acrt_d[np.newaxis, :] + xi @ cov_if["if_acrt_d"]
                att_glob_b = info["att_glob"] + xi @ cov_if["if_att_glob"]
                acrt_glob_b = xi @ cov_if["if_acrt_glob"]
                return att_d_b, acrt_d_b, att_glob_b, acrt_glob_b, info.get("acrt_glob", 0.0)
            treated_idx = info["treated_indices"]
            control_idx = info["control_indices"]
            n_t = info["n_treated"]
            n_c = info["n_control"]
            # Use survey-weighted masses when available (matching analytical SE)
            if "w_treated" in info:
                n_t = info["w_treated"]
                n_c = info["w_control"]
            bread = info["bread"]
            ee_treated = info["ee_treated"]
            ee_control = info["ee_control"]
            psi_bar = info["psi_bar"]
            beta_pred = info["beta_pred"]
            Psi_eval = info["Psi_eval"]
            dPsi_eval = info["dPsi_eval"]
            dPsi_treated = info["dPsi_treated"]
            delta_y_treated = info["delta_y_treated"]
            mu_0 = info["mu_0"]
            att_glob_gt = info["att_glob"]
            sw_treated = info.get("w_treated_arr")

            w_treated = all_weights[:, treated_idx]
            w_control = all_weights[:, control_idx]

            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                treated_sum = w_treated @ ee_treated / n_t
                control_sum = (w_control @ ee_control) / n_c
                psi_bar_outer = psi_bar[np.newaxis, :]

                delta_beta = (treated_sum - control_sum[:, np.newaxis] * psi_bar_outer) @ bread.T
                beta_b = beta_pred[np.newaxis, :] + delta_beta

                att_d_b = beta_b @ Psi_eval.T
                acrt_d_b = beta_b @ dPsi_eval.T

                mu_0_pert = (w_control @ ee_control) / n_c
                # ATT_glob perturbation: weight scores by survey weight w_k
                # when present, matching the analytical IF path.
                att_glob_score = delta_y_treated - att_glob_gt - mu_0
                if sw_treated is not None:
                    att_glob_score = sw_treated * att_glob_score
                mean_dy_treated_pert = (w_treated @ att_glob_score) / n_t
                att_glob_b = att_glob_gt + mean_dy_treated_pert - mu_0_pert

                if sw_treated is not None:
                    sw_norm = sw_treated / sw_treated.sum()
                    dpsi_mean = sw_norm @ dPsi_treated
                else:
                    dpsi_mean = np.mean(dPsi_treated, axis=0)
                acrt_glob_b = delta_beta @ dpsi_mean

            return att_d_b, acrt_d_b, att_glob_b, acrt_glob_b, info.get("acrt_glob", 0.0)

        # Iterate over post-treatment cells for dose-response/overall aggregation
        for gt, w in cell_weights.items():
            if w == 0:
                continue
            info = gt_bootstrap_info[gt]
            if not info:
                continue

            att_d_b, acrt_d_b, att_glob_b, acrt_glob_b, acrt_glob_pt = _bootstrap_gt_cell(gt, info)

            boot_att_d += w * att_d_b
            boot_acrt_d += w * acrt_d_b
            boot_att_glob += w * att_glob_b
            boot_acrt_glob += w * (acrt_glob_pt + acrt_glob_b)

        # Event study bootstrap — iterate over ALL (g,t) cells
        if event_study_effects is not None:
            for gt, r in gt_results.items():
                info = gt_bootstrap_info[gt]
                if not info:
                    continue
                g_val, t_val = gt
                e = t_val - g_val
                if e not in boot_es:
                    continue
                es_w = es_cell_weights.get(gt, 0.0)
                if es_w == 0:
                    continue
                _, _, att_glob_b, _, _ = _bootstrap_gt_cell(gt, info)
                boot_es[e] += es_w * att_glob_b

        # Compute statistics
        result: Dict[str, Any] = {}

        # Per-grid-point
        att_d_se = np.full(n_grid, np.nan)
        att_d_ci_lower = np.full(n_grid, np.nan)
        att_d_ci_upper = np.full(n_grid, np.nan)
        acrt_d_se = np.full(n_grid, np.nan)
        acrt_d_ci_lower = np.full(n_grid, np.nan)
        acrt_d_ci_upper = np.full(n_grid, np.nan)

        att_d_p = np.full(n_grid, np.nan)
        acrt_d_p = np.full(n_grid, np.nan)

        for idx in range(n_grid):
            se, ci, p = compute_effect_bootstrap_stats(
                original_att_d[idx],
                boot_att_d[:, idx],
                alpha=self.alpha,
                context=f"ATT(d) at grid point {idx}",
            )
            att_d_se[idx] = se
            att_d_ci_lower[idx] = ci[0]
            att_d_ci_upper[idx] = ci[1]
            att_d_p[idx] = p

            se, ci, p = compute_effect_bootstrap_stats(
                original_acrt_d[idx],
                boot_acrt_d[:, idx],
                alpha=self.alpha,
                context=f"ACRT(d) at grid point {idx}",
            )
            acrt_d_se[idx] = se
            acrt_d_ci_lower[idx] = ci[0]
            acrt_d_ci_upper[idx] = ci[1]
            acrt_d_p[idx] = p

        result["att_d_se"] = att_d_se
        result["att_d_ci_lower"] = att_d_ci_lower
        result["att_d_ci_upper"] = att_d_ci_upper
        result["acrt_d_se"] = acrt_d_se
        result["acrt_d_ci_lower"] = acrt_d_ci_lower
        result["acrt_d_ci_upper"] = acrt_d_ci_upper
        result["att_d_p"] = att_d_p
        result["acrt_d_p"] = acrt_d_p

        # Overall
        se, ci, p = compute_effect_bootstrap_stats(
            original_att,
            boot_att_glob,
            alpha=self.alpha,
            context="overall ATT_glob",
        )
        result["overall_att_se"] = se
        result["overall_att_ci"] = ci
        result["overall_att_p"] = p

        se, ci, p = compute_effect_bootstrap_stats(
            original_acrt,
            boot_acrt_glob,
            alpha=self.alpha,
            context="overall ACRT_glob",
        )
        result["overall_acrt_se"] = se
        result["overall_acrt_ci"] = ci
        result["overall_acrt_p"] = p

        # Event study SEs
        if event_study_effects is not None:
            es_se = {}
            es_ci = {}
            es_p = {}
            for e in es_keys:
                se_e, ci_e, p_e = compute_effect_bootstrap_stats(
                    event_study_effects[e]["effect"],
                    boot_es[e],
                    alpha=self.alpha,
                    context=f"event study e={e}",
                )
                es_se[e] = se_e
                es_ci[e] = ci_e
                es_p[e] = p_e
            result["es_se"] = es_se
            result["es_ci"] = es_ci
            result["es_p"] = es_p

        return result
