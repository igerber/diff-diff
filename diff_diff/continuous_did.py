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
from diff_diff.linalg import solve_ols
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
        self._validate_constrained_params()

    def _validate_constrained_params(self) -> None:
        """Validate control_group and base_period values."""
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
        }

    def set_params(self, **params) -> "ContinuousDiD":
        """Set estimator parameters and return self."""
        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(f"Invalid parameter: {key}")
            setattr(self, key, value)
        self._validate_constrained_params()
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

        Returns
        -------
        ContinuousDiDResults
        """
        # 1. Validate & prepare
        _VALID_AGGREGATES = (None, "dose", "eventstudy")
        if aggregate not in _VALID_AGGREGATES:
            raise ValueError(
                f"Invalid aggregate: '{aggregate}'. "
                f"Must be one of {_VALID_AGGREGATES}."
            )

        df = data.copy()
        for col in [outcome, unit, time, first_treat, dose]:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in data.")

        # Verify dose is time-invariant
        dose_nunique = df.groupby(unit)[dose].nunique()
        if dose_nunique.max() > 1:
            bad_units = dose_nunique[dose_nunique > 1].index.tolist()
            raise ValueError(
                f"Dose must be time-invariant. Units with varying dose: {bad_units[:5]}"
            )

        # Normalize first_treat: inf → 0
        df[first_treat] = df[first_treat].replace([np.inf, float("inf")], 0)

        # Drop units with positive first_treat but zero dose (R convention)
        unit_info = df.groupby(unit).first()[[first_treat, dose]]
        drop_units = unit_info[
            (unit_info[first_treat] > 0) & (unit_info[dose] == 0)
        ].index
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

        # Force dose=0 for never-treated units with nonzero dose
        never_treated_mask = df[first_treat] == 0
        if (df.loc[never_treated_mask, dose] != 0).any():
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

        # 2. Precompute structures
        precomp = self._precompute_structures(
            df, outcome, unit, time, first_treat, dose, time_periods
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
                    precomp, g, t, knots, degree, dvals
                )
                if result is not None:
                    gt_results[(g, t)] = result
                    gt_bootstrap_info[(g, t)] = result.get("_bootstrap_info", {})

        if len(gt_results) == 0:
            raise ValueError("No valid (g,t) cells computed.")

        # 4. Aggregate
        post_gt = {
            (g, t): r
            for (g, t), r in gt_results.items()
            if t >= g - self.anticipation
        }

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
            event_study_effects = self._aggregate_event_study(gt_results)

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
            # Within each group, weight is divided equally among post-treatment cells.
            group_n_treated = {}
            group_n_post_cells = {}
            for (g, t), r in post_gt.items():
                if g not in group_n_treated:
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
                    precomp, gt_results, gt_bootstrap_info, post_gt, cell_weights,
                    knots, degree, dvals, overall_att, overall_acrt,
                    agg_att_d, agg_acrt_d,
                    event_study_effects,
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
                overall_att_t = safe_inference(
                    overall_att, overall_att_se, self.alpha
                )[0]
                overall_att_p = boot_result["overall_att_p"]
                overall_att_ci = boot_result["overall_att_ci"]
                overall_acrt_se = boot_result["overall_acrt_se"]
                overall_acrt_t = safe_inference(
                    overall_acrt, overall_acrt_se, self.alpha
                )[0]
                overall_acrt_p = boot_result["overall_acrt_p"]
                overall_acrt_ci = boot_result["overall_acrt_ci"]
                if event_study_effects is not None:
                    for e, info in event_study_effects.items():
                        if e in boot_result.get("es_se", {}):
                            info["se"] = boot_result["es_se"][e]
                            info["t_stat"] = safe_inference(
                                info["effect"], info["se"], self.alpha
                            )[0]
                            info["p_value"] = boot_result["es_p"][e]
                            info["conf_int"] = boot_result["es_ci"][e]
            else:
                # Analytical SEs via influence functions
                analytic = self._compute_analytical_se(
                    precomp, gt_results, gt_bootstrap_info, post_gt, cell_weights,
                    knots, degree, dvals, agg_att_d, agg_acrt_d,
                )
                att_d_se = analytic["att_d_se"]
                acrt_d_se = analytic["acrt_d_se"]
                overall_att_se = analytic["overall_att_se"]
                overall_acrt_se = analytic["overall_acrt_se"]

                overall_att_t, overall_att_p, overall_att_ci = safe_inference(
                    overall_att, overall_att_se, self.alpha
                )
                overall_acrt_t, overall_acrt_p, overall_acrt_ci = safe_inference(
                    overall_acrt, overall_acrt_se, self.alpha
                )

                # Per-grid-point inference for dose-response
                for idx in range(n_grid):
                    _, _, ci = safe_inference(
                        agg_att_d[idx], att_d_se[idx], self.alpha
                    )
                    att_d_ci_lower[idx] = ci[0]
                    att_d_ci_upper[idx] = ci[1]

                    _, _, ci = safe_inference(
                        agg_acrt_d[idx], acrt_d_se[idx], self.alpha
                    )
                    acrt_d_ci_lower[idx] = ci[0]
                    acrt_d_ci_upper[idx] = ci[1]

                # Event study analytical SEs
                if event_study_effects is not None:
                    n_units = precomp["n_units"]
                    for e_val, info_e in event_study_effects.items():
                        # Collect (g,t) cells for this event-time bin
                        e_gts = [gt for gt in gt_results if gt[1] - gt[0] == e_val]
                        if not e_gts:
                            continue
                        # n_treated-proportional weights within this bin
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
                            treated_idx = b_info["treated_indices"]
                            control_idx = b_info["control_indices"]
                            n_t = b_info["n_treated"]
                            n_c = b_info["n_control"]
                            n_total_gt = n_t + n_c
                            p_1 = n_t / n_total_gt
                            p_0 = n_c / n_total_gt
                            att_glob_gt = b_info["att_glob"]
                            mu_0 = b_info["mu_0"]
                            delta_y_treated = b_info["delta_y_treated"]
                            ee_control = b_info["ee_control"]

                            for k, uid in enumerate(treated_idx):
                                if_es[uid] += (
                                    w
                                    * (delta_y_treated[k] - att_glob_gt - mu_0)
                                    / p_1
                                    / n_total_gt
                                )
                            for k, uid in enumerate(control_idx):
                                if_es[uid] -= (
                                    w * ee_control[k] / p_0 / n_total_gt
                                )

                        es_se = float(np.sqrt(np.sum(if_es**2)))
                        t_stat, p_val, ci_es = safe_inference(
                            info_e["effect"], es_se, self.alpha
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
        )

        # Strip bootstrap internals from gt_results
        clean_gt = {}
        for gt, r in gt_results.items():
            clean_gt[gt] = {
                k: v for k, v in r.items() if not k.startswith("_")
            }

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
            degree=self.degree,
            num_knots=self.num_knots,
            base_period=self.base_period,
            anticipation=self.anticipation,
            n_bootstrap=self.n_bootstrap,
            bootstrap_weights=self.bootstrap_weights,
            seed=self.seed,
            rank_deficient_action=self.rank_deficient_action,
            event_study_effects=event_study_effects,
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

        # Per-unit cohort and dose
        unit_cohorts = np.zeros(n_units, dtype=float)
        dose_vector = np.zeros(n_units, dtype=float)
        unit_first = df.groupby(unit).first()
        for u in all_units:
            i = unit_to_idx[u]
            unit_cohorts[i] = unit_first.loc[u, first_treat]
            dose_vector[i] = unit_first.loc[u, dose]

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
        }

    def _compute_dose_response_gt(
        self,
        precomp: Dict[str, Any],
        g: Any,
        t: Any,
        knots: np.ndarray,
        degree: int,
        dvals: np.ndarray,
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
        delta_y_treated = outcome_matrix[treated_mask, col_t] - outcome_matrix[treated_mask, col_base]
        delta_y_control = outcome_matrix[control_mask, col_t] - outcome_matrix[control_mask, col_base]

        # Control counterfactual
        mu_0 = float(np.mean(delta_y_control))

        # Demean
        delta_tilde_y = delta_y_treated - mu_0

        # Treated doses
        treated_doses = dose_vector[treated_mask]

        # B-spline OLS
        Psi = bspline_design_matrix(treated_doses, knots, degree, include_intercept=True)
        n_basis = Psi.shape[1]

        # Check for all-same dose
        if np.all(treated_doses == treated_doses[0]):
            warnings.warn(
                f"All treated doses identical in (g={g}, t={t}). "
                "ACRT(d) will be 0 everywhere.",
                UserWarning,
                stacklevel=3,
            )

        # Skip if not enough treated units for OLS (need n > K for residual df)
        if n_treated <= n_basis:
            warnings.warn(
                f"Not enough treated units ({n_treated}) for {n_basis} basis functions "
                f"in (g={g}, t={t}). Skipping cell.",
                UserWarning,
                stacklevel=3,
            )
            return None

        # OLS regression
        beta_hat, residuals, _ = solve_ols(
            Psi, delta_tilde_y,
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

        # Summary parameters
        att_glob = float(np.mean(delta_y_treated) - mu_0)

        # ACRT^{glob}: plug-in average of ACRT(D_i) for treated
        dPsi_treated = bspline_derivative_design_matrix(
            treated_doses, knots, degree, include_intercept=True
        )
        acrt_glob = float(np.mean(dPsi_treated @ beta_pred))

        # Store bootstrap info for influence function computation
        # bread = (Psi'Psi / n_treated)^{-1}
        PtP = Psi.T @ Psi
        try:
            bread = np.linalg.inv(PtP / n_treated)
        except np.linalg.LinAlgError:
            bread = np.linalg.pinv(PtP / n_treated)

        # ee_treated: per-unit estimating equation vectors (K-vector per unit)
        ee_treated = Psi * residuals[:, np.newaxis]  # (n_treated, K)

        # ee_control: per-unit deviation from control mean
        ee_control = delta_y_control - mu_0  # (n_control,)

        # psi_bar: mean basis vector for treated
        psi_bar = np.mean(Psi, axis=0)  # (K,)

        # Unit indices for bootstrap
        treated_indices = np.where(treated_mask)[0]
        control_indices = np.where(control_mask)[0]

        bootstrap_info = {
            "bread": bread,
            "ee_treated": ee_treated,
            "ee_control": ee_control,
            "psi_bar": psi_bar,
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
        }

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
    ) -> Dict[int, Dict[str, Any]]:
        """Aggregate binarized ATT_glob by relative period."""
        effects_by_e: Dict[int, List[Tuple[float, float]]] = {}

        for (g, t), r in gt_results.items():
            e = t - g
            if e not in effects_by_e:
                effects_by_e[e] = []
            effects_by_e[e].append((r["att_glob"], float(r["n_treated"])))

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
            treated_idx = info["treated_indices"]
            control_idx = info["control_indices"]
            n_t = info["n_treated"]
            n_c = info["n_control"]
            bread = info["bread"]
            ee_treated = info["ee_treated"]
            ee_control = info["ee_control"]
            psi_bar = info["psi_bar"]
            Psi_eval = info["Psi_eval"]
            dPsi_eval = info["dPsi_eval"]
            dPsi_treated = info["dPsi_treated"]
            att_glob_gt = info["att_glob"]
            mu_0 = info["mu_0"]
            delta_y_treated = info["delta_y_treated"]

            n_total = n_t + n_c
            p_1 = n_t / n_total
            p_0 = n_c / n_total

            # IF for ATT_glob (binarized DiD)
            for k, idx in enumerate(treated_idx):
                if_att_glob[idx] += w * (delta_y_treated[k] - att_glob_gt - mu_0) / p_1 / n_total
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
            dpsi_bar = np.mean(dPsi_treated, axis=0)
            for k, idx in enumerate(treated_idx):
                beta_pert = bread @ ee_treated[k] / n_t
                if_acrt_glob[idx] += w * float(dpsi_bar @ beta_pert)
            for k, idx in enumerate(control_idx):
                beta_pert = -bread @ psi_bar * ee_control[k] / n_c
                if_acrt_glob[idx] += w * float(dpsi_bar @ beta_pert)

        # SE = sqrt(sum(IF_i^2)), matching CallawaySantAnna's convention
        # (per-unit IFs already contain 1/n_t, 1/n_c scaling)
        overall_att_se = float(np.sqrt(np.sum(if_att_glob**2)))
        overall_acrt_se = float(np.sqrt(np.sum(if_acrt_glob**2)))

        att_d_se = np.sqrt(np.sum(if_att_d**2, axis=0))
        acrt_d_se = np.sqrt(np.sum(if_acrt_d**2, axis=0))

        return {
            "overall_att_se": overall_att_se,
            "overall_acrt_se": overall_acrt_se,
            "att_d_se": att_d_se,
            "acrt_d_se": acrt_d_se,
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
    ) -> Dict[str, Any]:
        """Run multiplier bootstrap inference."""
        if self.n_bootstrap < 50:
            warnings.warn(
                f"n_bootstrap={self.n_bootstrap} is low. Consider n_bootstrap >= 199 "
                "for reliable inference.",
                UserWarning,
                stacklevel=3,
            )

        rng = np.random.default_rng(self.seed)
        n_units = precomp["n_units"]
        n_grid = len(dvals)

        # Generate all weights upfront
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
        # Per-(g,t) weight within event-time bin
        es_cell_weights: Dict[Tuple, float] = {}
        if event_study_effects is not None:
            # Build event-time bin weights from n_treated
            from collections import defaultdict
            es_bin_total: Dict[int, float] = defaultdict(float)
            for gt, r in gt_results.items():
                g_val, t_val = gt
                e = t_val - g_val
                es_bin_total[e] += float(r["n_treated"])
            for gt, r in gt_results.items():
                g_val, t_val = gt
                e = t_val - g_val
                if es_bin_total[e] > 0:
                    es_cell_weights[gt] = float(r["n_treated"]) / es_bin_total[e]

        # Helper to bootstrap a single (g,t) cell
        def _bootstrap_gt_cell(gt, info):
            """Returns att_glob_b array (B,) for this cell."""
            treated_idx = info["treated_indices"]
            control_idx = info["control_indices"]
            n_t = info["n_treated"]
            n_c = info["n_control"]
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

            w_treated = all_weights[:, treated_idx]
            w_control = all_weights[:, control_idx]

            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                treated_sum = w_treated @ ee_treated / n_t
                control_sum = (w_control @ ee_control) / n_c
                psi_bar_outer = psi_bar[np.newaxis, :]

                delta_beta = (treated_sum - control_sum[:, np.newaxis] * psi_bar_outer) @ bread.T
                beta_b = beta_pred[np.newaxis, :] + delta_beta

                att_d_b = beta_b @ Psi_eval.T
                acrt_d_b = beta_b @ dPsi_eval.T

                mu_0_pert = (w_control @ ee_control) / n_c
                mean_dy_treated_pert = (w_treated @ (delta_y_treated - att_glob_gt - mu_0)) / n_t
                att_glob_b = att_glob_gt + mean_dy_treated_pert - mu_0_pert

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
                original_att_d[idx], boot_att_d[:, idx],
                alpha=self.alpha, context=f"ATT(d) at grid point {idx}",
            )
            att_d_se[idx] = se
            att_d_ci_lower[idx] = ci[0]
            att_d_ci_upper[idx] = ci[1]
            att_d_p[idx] = p

            se, ci, p = compute_effect_bootstrap_stats(
                original_acrt_d[idx], boot_acrt_d[:, idx],
                alpha=self.alpha, context=f"ACRT(d) at grid point {idx}",
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
            original_att, boot_att_glob, alpha=self.alpha, context="overall ATT_glob",
        )
        result["overall_att_se"] = se
        result["overall_att_ci"] = ci
        result["overall_att_p"] = p

        se, ci, p = compute_effect_bootstrap_stats(
            original_acrt, boot_acrt_glob, alpha=self.alpha, context="overall ACRT_glob",
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
                    event_study_effects[e]["effect"], boot_es[e],
                    alpha=self.alpha, context=f"event study e={e}",
                )
                es_se[e] = se_e
                es_ci[e] = ci_e
                es_p[e] = p_e
            result["es_se"] = es_se
            result["es_ci"] = es_ci
            result["es_p"] = es_p

        return result
