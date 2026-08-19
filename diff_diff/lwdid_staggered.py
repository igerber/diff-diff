"""Cohort-time estimation and joint aggregation for LWDiD."""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from diff_diff.lwdid_results import LWDiDResults
from diff_diff.utils import safe_inference

CellKey = Tuple[Any, Any]


def _guard_standard_error(effect: float, se: float) -> float:
    """Return NaN for numerically degenerate finite standard errors."""
    tolerance = np.sqrt(np.finfo(float).eps) * max(1.0, abs(effect))
    if not np.isfinite(se) or se <= tolerance:
        return np.nan
    return float(se)


def _effective_influence(
    influence: np.ndarray,
    cluster_ids: Optional[np.ndarray],
) -> np.ndarray:
    if cluster_ids is None:
        return influence
    frame = pd.DataFrame({"cluster": cluster_ids})
    columns = []
    for index in range(influence.shape[1]):
        frame["value"] = influence[:, index]
        columns.append(frame.groupby("cluster", sort=False)["value"].sum().to_numpy())
    return np.column_stack(columns)


def _combine_influence(
    keys: List[CellKey],
    weights: np.ndarray,
    cell_influence: Dict[CellKey, np.ndarray],
    n_units: int,
) -> Optional[np.ndarray]:
    if any(key not in cell_influence for key in keys):
        return None
    combined = np.zeros(n_units, dtype=float)
    for key, weight in zip(keys, weights):
        combined += float(weight) * cell_influence[key]
    return combined


def _inference_from_influence(
    effect: float,
    influence: Optional[np.ndarray],
    alpha: float,
    cluster_ids: Optional[np.ndarray],
    *,
    df_unclustered: Optional[int] = None,
    contributing_mask: Optional[np.ndarray] = None,
) -> Tuple[float, float, float, Tuple[float, float], Optional[int]]:
    """Aggregate-level inference from a combined influence vector.

    Reference-distribution policy (fix-wave WS6): clustered aggregates use
    ``G - 1`` where G counts the clusters CONTRIBUTING to the aggregate
    (``contributing_mask``; clusters supplying no estimated cell must not
    inflate the df); unclustered aggregates composed of EXACTLY ONE cell
    use that cell's residual df (``df_unclustered`` - matching the
    common-timing rules, so a single-post-period staggered fit and the
    common-timing fit of the same data agree); unclustered multi-cell
    aggregates keep the large-sample normal reference (units recur across
    cells with overlapping influence functions, so no residual-df pooling
    is valid - documented in REGISTRY).
    """
    if influence is None:
        return np.nan, np.nan, np.nan, (np.nan, np.nan), None
    effective = _effective_influence(influence[:, None], cluster_ids)[:, 0]
    se = _guard_standard_error(effect, float(np.sqrt(np.sum(effective**2))))
    if not np.isfinite(se):
        return np.nan, np.nan, np.nan, (np.nan, np.nan), None
    if cluster_ids is not None:
        ids = cluster_ids if contributing_mask is None else cluster_ids[contributing_mask]
        df: Optional[int] = max(len(np.unique(ids)) - 1, 1)
    else:
        df = df_unclustered
    t_stat, p_value, conf_int = safe_inference(effect, se, alpha=alpha, df=df)
    return se, t_stat, p_value, conf_int, df


def _empty_cell(
    g: Any,
    t: Any,
    reason: str,
    n_treated: int = 0,
    n_control: int = 0,
) -> Dict[str, Any]:
    return {
        "cohort": g,
        "time": t,
        "relative_time": t - g,
        "att": np.nan,
        "se": np.nan,
        "t_stat": np.nan,
        "p_value": np.nan,
        "conf_int": (np.nan, np.nan),
        "n_treated": n_treated,
        "n_control": n_control,
        "df": None,
        "skip_reason": reason,
        "inference_status": "not_estimable",
    }


def _transform_for_cohort(
    estimator: Any,
    frame: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    g: Any,
) -> pd.DataFrame:
    pre_mask = frame[time] < g
    if estimator.rolling == "demean":
        return estimator._transform_demean(frame, outcome, unit, pre_mask)
    if estimator.rolling == "detrend":
        return estimator._transform_detrend(frame, outcome, unit, time, pre_mask)
    if estimator.rolling == "demeanq":
        return estimator._transform_demeanq(frame, outcome, unit, time, pre_mask)
    return estimator._transform_detrendq(frame, outcome, unit, time, pre_mask)


def compute_event_study_bands(
    estimator: Any,
    event_effects: Dict[int, Dict[str, Any]],
    event_influence: Dict[int, np.ndarray],
    cluster_ids: Optional[np.ndarray],
) -> Tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[str],
    Optional[float],
    Optional[int],
]:
    """Analytical event-study covariance plus optional multiplier bootstrap.

    Shared by the staggered and common-timing paths. The analytical
    covariance of the event-study effects is the cross-product of their
    effective (cluster-summed) influence columns. When
    ``estimator.n_bootstrap > 0`` the Rademacher multiplier bootstrap
    replaces the analytical per-event SEs in ``event_effects`` in place,
    attaches sup-t simultaneous ``cband_conf_int`` bounds, and suppresses
    the (now inconsistent) analytical covariance.

    Returns
    -------
    tuple
        ``(event_vcov, event_vcov_index, cband_method, cband_crit_value,
        cband_n_bootstrap)``.
    """
    event_labels = sorted(event_influence)
    event_vcov = None
    event_vcov_index = None
    cband_method = None
    cband_crit_value = None
    cband_n_bootstrap = None
    if event_labels:
        influence_matrix = np.column_stack([event_influence[label] for label in event_labels])
        effective = _effective_influence(influence_matrix, cluster_ids)
        event_vcov = effective.T @ effective
        event_vcov_index = np.array(event_labels)
        if estimator.n_bootstrap > 0:
            rng = np.random.default_rng(estimator.seed)
            centered = effective - effective.mean(axis=0, keepdims=True)
            multipliers = rng.choice([-1.0, 1.0], size=(estimator.n_bootstrap, centered.shape[0]))
            draws = multipliers @ centered
            bootstrap_se = np.std(draws, axis=0, ddof=1)
            valid = np.isfinite(bootstrap_se) & (bootstrap_se > 0)
            if valid.any():
                sup_t = np.max(np.abs(draws[:, valid]) / bootstrap_se[valid], axis=1)
                cband_crit_value = float(np.quantile(sup_t, 1 - estimator.alpha))
                cband_method = "multiplier_bootstrap_sup_t"
                cband_n_bootstrap = estimator.n_bootstrap
                for index, label in enumerate(event_labels):
                    if not valid[index]:
                        continue
                    row = event_effects[label]
                    row["se"] = float(bootstrap_se[index])
                    row["t_stat"], row["p_value"], row["conf_int"] = safe_inference(
                        row["effect"], row["se"], alpha=estimator.alpha, df=row.get("df")
                    )
                    row["cband_conf_int"] = (
                        row["effect"] - cband_crit_value * row["se"],
                        row["effect"] + cband_crit_value * row["se"],
                    )
            # Bootstrap SEs replace the analytical diagonal, so do not expose
            # an inconsistent analytical covariance matrix.
            event_vcov = None
            event_vcov_index = None
    return event_vcov, event_vcov_index, cband_method, cband_crit_value, cband_n_bootstrap


def fit_staggered(
    estimator: Any,
    df: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    cohort: str,
    cluster: Optional[str],
    controls: List[str],
) -> LWDiDResults:
    """Estimate all supported cohort-time cells and aggregate them jointly."""
    varying = df.groupby(unit)[cohort].nunique(dropna=False)
    if (varying > 1).any():
        raise ValueError(
            f"Cohort must be time-invariant. Found {int((varying > 1).sum())} "
            "unit(s) with varying cohort."
        )
    # Unit-constancy of covariate and cluster columns is enforced by the
    # shared fit() validation layer (LWDiD._validate_inputs), which covers
    # this path and the common-timing path alike.

    unit_rows = df.drop_duplicates(subset=[unit], keep="first").set_index(unit)
    all_units = unit_rows.index.to_list()
    unit_to_index = {value: index for index, value in enumerate(all_units)}
    cohort_by_unit = unit_rows[cohort]
    never_mask = cohort_by_unit.isna() | (cohort_by_unit == 0)
    never_units = cohort_by_unit.index[never_mask].to_list()
    treated_cohorts = sorted(
        value for value in pd.unique(df[cohort]) if pd.notna(value) and value > 0
    )
    if not treated_cohorts:
        raise ValueError("No treated cohorts found.")
    if estimator.control_group == "never_treated" and len(never_units) < 2:
        raise ValueError(
            "control_group='never_treated' requires at least 2 never-treated "
            f"units for valid estimation; found {len(never_units)}."
        )
    if estimator.control_group == "not_yet_treated" and not never_units:
        raise ValueError(
            "All units are eventually treated: control_group='not_yet_treated' "
            "requires at least one never-treated unit (or an explicit "
            "reference cohort, which is not supported). Without one, the "
            "latest cohort-time cells have no valid control group and the "
            "estimand would be silently truncated."
        )

    all_times = sorted(pd.unique(df[time]))
    reference_periods = (-1,) if estimator.rolling in ("demean", "demeanq") else (-2, -1)
    global_cluster_ids = None
    if cluster is not None:
        if cluster == unit:
            # The unit column was consumed by set_index; read it from the index.
            global_cluster_ids = unit_rows.index.to_numpy()
        else:
            global_cluster_ids = unit_rows.loc[all_units, cluster].to_numpy()

    cell_effects: Dict[CellKey, Dict[str, Any]] = {}
    cell_influence: Dict[CellKey, np.ndarray] = {}
    cell_members: Dict[CellKey, np.ndarray] = {}
    single_cluster_cells: List[CellKey] = []
    skipped: List[Tuple[Any, Any, str]] = []
    cohort_sizes: Dict[Any, int] = {}

    for g in treated_cohorts:
        treated_units = cohort_by_unit.index[cohort_by_unit == g].to_list()
        cohort_sizes[g] = len(treated_units)
        if estimator.control_group == "never_treated":
            control_superset = never_units
        else:
            later = cohort_by_unit.index[cohort_by_unit > g].to_list()
            control_superset = never_units + later
        relevant_units = list(dict.fromkeys(treated_units + control_superset))
        cohort_frame = df.loc[df[unit].isin(relevant_units)].copy()
        n_pre_periods = len([value for value in all_times if value < g])
        required_pre = 2 if estimator.rolling in ("detrend", "detrendq") else 1
        if n_pre_periods < required_pre:
            for t in all_times:
                if (t - g) not in reference_periods:
                    key = (g, t)
                    cell_effects[key] = _empty_cell(g, t, "insufficient_pre_periods")
                    skipped.append((g, t, "insufficient_pre_periods"))
            continue

        transformed = _transform_for_cohort(estimator, cohort_frame, outcome, unit, time, g)
        for t in all_times:
            relative_time = t - g
            if relative_time in reference_periods:
                continue
            key = (g, t)
            if estimator.control_group == "never_treated":
                valid_controls = set(never_units)
            else:
                threshold = max(g, t)
                valid_controls = set(never_units)
                valid_controls.update(cohort_by_unit.index[cohort_by_unit > threshold].to_list())

            sample_units = set(treated_units) | valid_controls
            columns = [unit, "_ydot"] + controls
            if cluster is not None and cluster not in columns:
                columns.append(cluster)
            cell = transformed.loc[
                (transformed[time] == t) & transformed[unit].isin(sample_units), columns
            ].drop_duplicates(subset=[unit], keep="first")
            finite = np.isfinite(cell["_ydot"].to_numpy(dtype=float))
            if controls:
                finite &= np.all(np.isfinite(cell[controls].to_numpy(dtype=float)), axis=1)
            cell = cell.loc[finite].copy()
            treatment = cell[unit].isin(treated_units).to_numpy(dtype=float)
            n_treated = int(treatment.sum())
            n_control = int(len(treatment) - n_treated)
            if n_treated == 0 or n_control == 0:
                cell_effects[key] = _empty_cell(g, t, "zero_treated_control", n_treated, n_control)
                skipped.append((g, t, "zero_treated_control"))
                continue

            y = cell["_ydot"].to_numpy(dtype=float)
            controls_matrix = cell[controls].to_numpy(dtype=float) if controls else None
            cluster_ids = None
            cell_single_cluster = False
            if cluster is not None:
                cluster_ids = cell[cluster].to_numpy()
                if len(np.unique(cluster_ids)) < 2:
                    # Cluster ids are re-derived per cell, so a cell whose
                    # units share one cluster can exist with G >= 2
                    # globally. Estimate the POINT unclustered and fail the
                    # inference closed below (campaign finding: ipw/dr
                    # silently fell back to unclustered variance here under
                    # a CR1 label; reg raised mid-fit).
                    cell_single_cluster = True
                    cluster_ids = None
            att, se, _, _, n_params, influence = estimator._dispatch_estimator(
                y, treatment, controls_matrix, cluster_ids, len(cell)
            )
            if not np.isfinite(att):
                cell_effects[key] = _empty_cell(g, t, "non_finite_estimate", n_treated, n_control)
                skipped.append((g, t, "non_finite_estimate"))
                continue

            se = _guard_standard_error(att, se)
            if cell_single_cluster:
                # Fail closed: point retained, inference NaN; aggregates
                # that include this cell inherit NaN inference (deliberate
                # - see the aggregated warning below and REGISTRY).
                single_cluster_cells.append(key)
                se = np.nan
                influence = None
            if cluster_ids is not None:
                df_cell = max(len(np.unique(cluster_ids)) - 1, 1)
            else:
                # n_params is the fitted design's parameter count, so the
                # residual df is design-coherent for every method.
                df_cell = max(len(cell) - n_params, 1)
            t_stat, p_value, conf_int = safe_inference(att, se, alpha=estimator.alpha, df=df_cell)
            cell_effects[key] = {
                "cohort": g,
                "time": t,
                "relative_time": relative_time,
                "att": float(att),
                "se": se,
                "t_stat": t_stat,
                "p_value": p_value,
                "conf_int": conf_int,
                "n_treated": n_treated,
                "n_control": n_control,
                "df": df_cell,
                "skip_reason": None,
                "inference_status": "ok" if np.isfinite(se) else "degenerate",
            }
            member_mask = np.zeros(len(all_units), dtype=bool)
            for unit_value in cell[unit].unique():
                member_mask[unit_to_index[unit_value]] = True
            cell_members[key] = member_mask
            if influence is not None and np.isfinite(se):
                global_influence = np.zeros(len(all_units), dtype=float)
                for local_index, unit_value in enumerate(cell[unit].to_list()):
                    global_influence[unit_to_index[unit_value]] = influence[local_index]
                cell_influence[key] = global_influence

    if single_cluster_cells:
        listed = ", ".join(str(k) for k in single_cluster_cells[:6])
        suffix = "" if len(single_cluster_cells) <= 6 else f"; plus {len(single_cluster_cells) - 6} more"
        warnings.warn(
            f"LWDiD: cohort-time cell(s) {listed}{suffix} contain fewer than "
            "2 clusters, so their cluster-robust inference is not identified. "
            "Cell points are retained with NaN inference; any aggregate that "
            "includes such a cell reports NaN inference as well (fail-closed).",
            UserWarning,
            stacklevel=2,
        )
    if skipped:
        preview = ", ".join(f"({g}, {t}): {reason}" for g, t, reason in skipped[:6])
        suffix = "" if len(skipped) <= 6 else f"; plus {len(skipped) - 6} more"
        warnings.warn(
            f"LWDiD skipped {len(skipped)} unsupported cohort-time cell(s): " f"{preview}{suffix}.",
            UserWarning,
            stacklevel=2,
        )

    cohort_effects: Dict[Any, Dict[str, Any]] = {}
    cohort_influence: Dict[Any, np.ndarray] = {}
    for g in treated_cohorts:
        keys = [
            key
            for key, value in cell_effects.items()
            if key[0] == g and key[1] >= g and np.isfinite(value["att"])
        ]
        if not keys:
            continue
        masses = np.array([cell_effects[key]["n_treated"] for key in keys], dtype=float)
        weights = masses / masses.sum()
        effect = float(np.dot(weights, [cell_effects[key]["att"] for key in keys]))
        influence = _combine_influence(keys, weights, cell_influence, len(all_units))
        mask = np.zeros(len(all_units), dtype=bool)
        for key in keys:
            mask |= cell_members.get(key, False)
        se, t_stat, p_value, conf_int, df_group = _inference_from_influence(
            effect,
            influence,
            estimator.alpha,
            global_cluster_ids,
            df_unclustered=(cell_effects[keys[0]]["df"] if len(keys) == 1 else None),
            contributing_mask=mask,
        )
        cohort_effects[g] = {
            "cohort": g,
            "att": effect,
            "se": se,
            "t_stat": t_stat,
            "p_value": p_value,
            "conf_int": conf_int,
            "n_treated": cohort_sizes[g],
            "n_control": max(cell_effects[key]["n_control"] for key in keys),
            "n_cells": len(keys),
            "df": df_group,
        }
        if influence is not None:
            cohort_influence[g] = influence

    if not cohort_effects:
        raise ValueError("No supported post-treatment cohort-time cells were estimable.")

    valid_cohorts = list(cohort_effects)
    overall_keys = [
        key
        for key, value in cell_effects.items()
        if key[0] in cohort_effects and key[1] >= key[0] and np.isfinite(value["att"])
    ]
    overall_cluster_mask = np.zeros(len(all_units), dtype=bool)
    for key in overall_keys:
        overall_cluster_mask |= cell_members.get(key, False)
    cohort_masses = np.array([cohort_sizes[g] for g in valid_cohorts], dtype=float)
    cohort_weights = cohort_masses / cohort_masses.sum()
    for g, weight in zip(valid_cohorts, cohort_weights):
        cohort_effects[g]["weight"] = float(weight)
    overall_effect = float(
        np.dot(cohort_weights, [cohort_effects[g]["att"] for g in valid_cohorts])
    )
    # LW 2026 (7.16)/(7.18): with never-treated controls, regression
    # adjustment and no covariates, the overall estimand is tau_omega --
    # the coefficient on D in the composite-outcome cross-sectional
    # regression, which averages each unit's transformed outcome over its
    # OBSERVED post periods. The composite is defined for the plain
    # demean/detrend transforms only (the q variants have no seasonal
    # composite; their overall is the cohort-mass-weighted average of
    # seasonal cohort ATTs).
    tau_omega_config = (
        estimator.control_group == "never_treated"
        and estimator.estimation_method == "reg"
        and not controls
        and estimator.rolling in ("demean", "detrend")
    )
    use_composite = (
        tau_omega_config
        and estimator.vcov_type == "classical"
        and cluster is None
    )
    # Complete-case resolution: the composite is computed ONCE for every
    # tau_omega-eligible configuration. With ZERO complete-case drops the
    # composite point is reported on BOTH vcov routes (status quo: the
    # classical route pairs it with the composite's own SE, hc1/clustered
    # with the joint-IF SE -- the documented approximation in REGISTRY).
    # With ANY drops, `.att` is the IF-weighted cohort-mass point on ALL
    # routes (the same point under every vcov setting -- a variance
    # selection must never move the point) and the complete-case composite
    # is exposed as the diagnostic `att_tau_omega_complete_case`.
    att_tau_omega_complete_case: Optional[float] = None
    n_composite_treated_dropped = 0
    n_composite_controls_dropped = 0
    composite_is_att = False
    comp_att = comp_se = np.nan
    comp_df = 0
    if tau_omega_config:
        (
            comp_att,
            comp_se,
            comp_df,
            n_composite_treated_dropped,
            n_composite_controls_dropped,
        ) = estimator._composite_regression_aggregation(df, outcome, unit, time, cohort)
        composite_drops = n_composite_treated_dropped + n_composite_controls_dropped
        if composite_drops == 0 and np.isfinite(comp_att):
            composite_is_att = True
        else:
            att_tau_omega_complete_case = float(comp_att) if np.isfinite(comp_att) else None
            warnings.warn(
                "LWDiD: the tau_omega composite required complete-case "
                f"drops ({n_composite_treated_dropped} treated, "
                f"{n_composite_controls_dropped} control unit(s)) on this "
                "unbalanced panel, so `.att` reports the influence-weighted "
                "cohort-mass point (identical under every vcov setting) "
                "instead of tau_omega. The complete-case composite is "
                "available as `att_tau_omega_complete_case`. See "
                "docs/methodology/REGISTRY.md (LWDiD).",
                UserWarning,
                stacklevel=2,
            )
    if composite_is_att:
        overall_effect = float(comp_att)
    if use_composite and composite_is_att:
        overall_se = _guard_standard_error(overall_effect, comp_se)
        overall_df = comp_df
        inference_basis = "composite_regression"
    else:
        overall_influence = None
        missing = [g for g in valid_cohorts if g not in cohort_influence]
        if not missing:
            overall_influence = sum(
                float(weight) * cohort_influence[g]
                for g, weight in zip(valid_cohorts, cohort_weights)
            )
        overall_se, _, _, _, overall_df = _inference_from_influence(
            overall_effect,
            overall_influence,
            estimator.alpha,
            global_cluster_ids,
            df_unclustered=(
                cell_effects[overall_keys[0]]["df"] if len(overall_keys) == 1 else None
            ),
            contributing_mask=overall_cluster_mask,
        )
        if overall_influence is not None:
            inference_basis = "joint_influence_function"
        elif estimator.estimation_method == "psm":
            inference_basis = "unavailable_matching"
            warnings.warn(
                "LWDiD: propensity-score matching has no influence-function "
                "representation, so cohort effects cannot be combined without "
                "assuming independence. Overall inference is reported as NaN; "
                "use estimation_method='dr' for a doubly robust alternative "
                "with valid joint inference.",
                UserWarning,
                stacklevel=2,
            )
        else:
            inference_basis = "unavailable_degenerate_cells"
            listed = ", ".join(str(g) for g in missing)
            warnings.warn(
                f"LWDiD: cohort(s) {listed} contain cohort-time cells with a "
                "degenerate or non-finite standard error, so no joint influence "
                "function is available. Overall inference is reported as NaN.",
                UserWarning,
                stacklevel=2,
            )
    overall_t, overall_p, overall_ci = safe_inference(
        overall_effect, overall_se, alpha=estimator.alpha, df=overall_df
    )

    event_effects: Dict[int, Dict[str, Any]] = {}
    event_influence: Dict[int, np.ndarray] = {}
    for relative_time in sorted({value["relative_time"] for value in cell_effects.values()}):
        keys = [
            key
            for key, value in cell_effects.items()
            if value["relative_time"] == relative_time and np.isfinite(value["att"])
        ]
        if not keys:
            continue
        masses = np.array([cell_effects[key]["n_treated"] for key in keys], dtype=float)
        weights = masses / masses.sum()
        effect = float(np.dot(weights, [cell_effects[key]["att"] for key in keys]))
        influence = _combine_influence(keys, weights, cell_influence, len(all_units))
        mask = np.zeros(len(all_units), dtype=bool)
        for key in keys:
            mask |= cell_members.get(key, False)
        se, t_stat, p_value, conf_int, df_event = _inference_from_influence(
            effect,
            influence,
            estimator.alpha,
            global_cluster_ids,
            df_unclustered=(cell_effects[keys[0]]["df"] if len(keys) == 1 else None),
            contributing_mask=mask,
        )
        event_effects[int(relative_time)] = {
            "effect": effect,
            "se": se,
            "t_stat": t_stat,
            "p_value": p_value,
            "conf_int": conf_int,
            "n_treated": int(masses.sum()),
            "n_cells": len(keys),
            "df": df_event,
        }
        if influence is not None:
            event_influence[int(relative_time)] = influence

    (
        event_vcov,
        event_vcov_index,
        cband_method,
        cband_crit_value,
        cband_n_bootstrap,
    ) = compute_event_study_bands(estimator, event_effects, event_influence, global_cluster_ids)

    # Sample metadata describes the CELL-ESTIMATION sample: units actually
    # contributing to at least one estimated cell (campaign finding: the
    # previous counts covered every input unit, overstating the sample
    # whenever cell-level drops occurred). Complete-case composite drops
    # are reported separately via n_composite_*_dropped.
    contributing_units_mask = np.zeros(len(all_units), dtype=bool)
    for member_mask in cell_members.values():
        contributing_units_mask |= member_mask
    contributing_index = np.flatnonzero(contributing_units_mask)
    contributing_ids = [all_units[i] for i in contributing_index]
    never_set = set(never_units)
    n_contrib_control = sum(1 for u in contributing_ids if u in never_set)
    n_contrib_treated = int((~never_mask.loc[contributing_ids]).sum())
    result = LWDiDResults(
        att=float(overall_effect),
        se=float(overall_se),
        t_stat=overall_t,
        p_value=overall_p,
        conf_int=overall_ci,
        n_obs=int(contributing_units_mask.sum()),
        n_treated=n_contrib_treated,
        n_control=n_contrib_control,
        rolling=estimator.rolling,
        estimation_method=estimator.estimation_method,
        vcov_type=estimator.vcov_type,
        alpha=estimator.alpha,
        df_inference=overall_df,
        cluster_name=cluster,
        n_clusters=(
            len(np.unique(global_cluster_ids[overall_cluster_mask]))
            if global_cluster_ids is not None
            else None
        ),
        cohort_effects=cohort_effects,
        cohort_time_effects=cell_effects,
        inference_basis=inference_basis,
        att_tau_omega_complete_case=att_tau_omega_complete_case,
        n_composite_treated_dropped=n_composite_treated_dropped,
        n_composite_controls_dropped=n_composite_controls_dropped,
        event_study_effects=event_effects,
        event_study_vcov=event_vcov,
        event_study_vcov_index=event_vcov_index,
        event_study_df={
            label: value["df"]
            for label, value in event_effects.items()
            if value.get("df") is not None
        },
        reference_periods=reference_periods,
        cband_method=cband_method,
        cband_crit_value=cband_crit_value,
        cband_n_bootstrap=cband_n_bootstrap,
    )
    return result
