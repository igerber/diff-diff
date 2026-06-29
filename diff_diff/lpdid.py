import warnings
from typing import Any, Dict, Iterable, Optional, Union

import numpy as np
import pandas as pd

from diff_diff.linalg import _rank_guarded_inv, solve_ols
from diff_diff.lpdid_results import LPDiDResults
from diff_diff.utils import safe_inference

__all__ = ["LPDiD", "LPDiDResults"]


class LPDiD:
    def __init__(
        self,
        pre_window: int = 2,
        post_window: int = 0,
        control_group: str = "clean",
        reweight: bool = False,
        no_composition: bool = False,
        pmd: Optional[Union[str, int]] = None,
        alpha: float = 0.05,
        cluster: Optional[str] = None,
        rank_deficient_action: str = "warn",
        non_absorbing: Optional[str] = None,
        stabilization_window: Optional[int] = None,
    ):
        self.pre_window = pre_window
        self.post_window = post_window
        self.control_group = control_group
        self.reweight = reweight
        self.no_composition = no_composition
        self.pmd = pmd
        self.alpha = alpha
        self.cluster = cluster
        self.rank_deficient_action = rank_deficient_action
        self.non_absorbing = non_absorbing
        self.stabilization_window = stabilization_window
        self._validate_params()
        self.is_fitted_ = False
        self.results_: Optional[LPDiDResults] = None

    def _validate_params(self) -> None:
        for _name, _val in (("pre_window", self.pre_window), ("post_window", self.post_window)):
            if not isinstance(_val, int) or isinstance(_val, bool) or _val < 0:
                raise ValueError(f"{_name} must be a non-negative integer")
        if (
            isinstance(self.alpha, bool)
            or not isinstance(self.alpha, (int, float))
            or not (0.0 < float(self.alpha) < 1.0)
        ):
            raise ValueError("alpha must be a float in (0, 1)")
        if self.control_group not in ("clean", "never_treated"):
            raise ValueError("control_group must be 'clean' or 'never_treated'")
        if self.rank_deficient_action not in ("warn", "error", "silent"):
            raise ValueError("rank_deficient_action must be 'warn', 'error', or 'silent'")
        if self.pmd is not None and not (
            self.pmd == "max"
            or (isinstance(self.pmd, int) and not isinstance(self.pmd, bool) and self.pmd > 0)
        ):
            raise ValueError("pmd must be None, 'max', or a positive integer")
        if self.non_absorbing not in (None, "first_entry", "effect_stabilization"):
            raise ValueError(
                "non_absorbing must be None (absorbing treatment), 'first_entry' "
                "(Dube et al. 2025 Eq. 12), or 'effect_stabilization' (Eq. 13)"
            )
        if self.non_absorbing == "effect_stabilization":
            if (
                isinstance(self.stabilization_window, bool)
                or not isinstance(self.stabilization_window, int)
                or self.stabilization_window < 1
            ):
                raise ValueError(
                    "stabilization_window (the paper's L) must be a positive integer when "
                    "non_absorbing='effect_stabilization'"
                )
        elif self.stabilization_window is not None:
            raise ValueError(
                "stabilization_window only applies when non_absorbing='effect_stabilization'; "
                "leave it None for absorbing or first_entry modes"
            )
        if self.non_absorbing is not None and self.control_group == "never_treated":
            raise ValueError(
                "control_group='never_treated' is not supported with a non-absorbing mode "
                "(the estimand becomes ambiguous); use control_group='clean' (the default)"
            )

    def _rhs_column_names(self, covariates=None, ylags=0, dylags=0):
        rhs_columns = list(covariates or [])
        rhs_columns.extend([f"_y_lag_{lag}" for lag in range(1, ylags + 1)])
        rhs_columns.extend([f"_dy_lag_{lag}" for lag in range(1, dylags + 1)])
        return rhs_columns

    def _prepare_panel(
        self,
        data,
        outcome,
        unit,
        time,
        treatment,
        cluster,
        covariates=None,
        ylags=0,
        dylags=0,
        absorb=None,
    ):
        selected_columns = list(
            dict.fromkeys(
                [unit, time, outcome, treatment, cluster, *(covariates or []), *(absorb or [])]
            )
        )
        panel = data[selected_columns].copy()
        panel = panel.sort_values([unit, time]).reset_index(drop=True)

        if panel.duplicated([unit, time]).any():
            raise ValueError("LPDiD requires unique unit-time observations")

        treated_numeric = pd.to_numeric(panel[treatment], errors="coerce")
        if treated_numeric.isna().any() or not treated_numeric.isin([0, 1]).all():
            raise ValueError(
                "treatment must contain binary numeric 0/1 values with no missing data"
            )

        panel["_treated"] = treated_numeric.astype(int)
        panel["_cluster"] = panel[cluster]
        if panel["_cluster"].isna().any():
            raise ValueError(
                f"cluster column '{cluster}' contains missing values; LPDiD cannot form "
                "cluster-robust standard errors with missing cluster labels (the affected "
                "rows would silently drop from the variance)."
            )

        # Absorbing-path validation and entry detection run on the OBSERVED rows,
        # BEFORE any calendar reindex below (the absorbing fill would otherwise make
        # the monotonicity check trivially pass, and gap rows carry NaN clusters).
        if self.non_absorbing is None:
            treated_cummax = panel.groupby(unit)["_treated"].cummax()
            if (treated_cummax > panel["_treated"]).any():
                raise ValueError(
                    "LPDiD requires an absorbing treatment path (once treated, always "
                    "treated) unless non_absorbing is set to 'first_entry' or "
                    "'effect_stabilization' (Dube et al. 2025 Section 4.2)"
                )
        # Entry = first OBSERVED treated period (documented convention; an unobserved
        # pre-onset gap is unknowable). For an absorbing path this is min(t | D=1).
        first_treat = panel.loc[panel["_treated"].eq(1)].groupby(unit)[time].min()

        # LP-DiD's per-unit features (outcome lags, first differences, premean
        # baselines) are CALENDAR quantities (t-1, t-k, t+h). Computing them with
        # row-order ops (shift/diff/rolling) silently equates "previous observed row"
        # with "calendar t-1", which is wrong when a unit has an interior time gap.
        # Reindex each unit to its complete interior calendar grid so every row-order
        # op is calendar-correct, compute the features on the grid, then restrict back
        # to the observed rows so the synthetic NaN gap rows never enter a regression.
        # A gap-free panel skips this entirely and is bit-identical to before.
        span = panel.groupby(unit)[time].agg(["min", "max", "nunique"])
        has_gap = bool((span["nunique"] != (span["max"] - span["min"] + 1)).any())
        if self.non_absorbing is not None and has_gap:
            raise ValueError(
                "LPDiD non-absorbing modes require gap-free panels within each unit's "
                "observed span: the [t-L, t+h] window conditions cannot be verified across "
                "an interior time gap. Fill or regularize the panel, or use the absorbing "
                "path. (Interior-gap support for non-absorbing treatment is a deferred "
                "follow-up.)"
            )
        if has_gap:
            panel["_observed"] = True
            grid = pd.concat(
                [
                    pd.DataFrame({unit: u, time: np.arange(int(lo), int(hi) + 1)})
                    for u, lo, hi in zip(span.index, span["min"], span["max"])
                ],
                ignore_index=True,
            )
            panel = grid.merge(panel, on=[unit, time], how="left")
            panel["_observed"] = panel["_observed"].fillna(False).astype(bool)
            panel = panel.sort_values([unit, time]).reset_index(drop=True)
            panel["_first_treat"] = panel[unit].map(first_treat).astype(float).fillna(np.inf)
            # Absorbing fill: treatment is fully determined by the entry period, so the
            # filled gap rows are consistent and observed rows are reproduced exactly.
            panel["_treated"] = (panel[time] >= panel["_first_treat"]).astype(int)
            panel["_entry"] = (panel[time] == panel["_first_treat"]).astype(float)
        else:
            panel["_first_treat"] = panel[unit].map(first_treat).astype(float).fillna(np.inf)
            panel["_entry"] = (panel[time] == panel["_first_treat"]).astype(float)

        if self.non_absorbing is not None:
            # Non-absorbing window operators (Dube et al. 2025 Section 4.2 / online
            # Appendix C). Non-absorbing requires a gap-free panel (enforced above), so
            # each unit's observed rows ARE its complete calendar grid and the groupby
            # shift/cumsum below are calendar-correct. `_treated` here is the GENUINE
            # treatment (the absorbing fill above is skipped for non-absorbing), so off
            # periods are preserved. Boundary convention (extends Deviation 5): periods
            # before a unit's first observed period are untreated with no change, so the
            # first first-difference uses fill_value=0 and the mask lookups clamp
            # pre-`_unit_min_t` offsets to 0.
            prev_treated = panel.groupby(unit)["_treated"].shift(1, fill_value=0)
            panel["_delta_d"] = panel["_treated"].astype(float) - prev_treated.astype(float)
            # `_switch_cum` = cumulative count of treatment CHANGES (for "no change in
            # window" conditions); `_d_cum` = cumulative SUM of D (for "D=0 across window"
            # level conditions). Both keyed by calendar time for offset lookups.
            panel["_switch_cum"] = panel["_delta_d"].abs().groupby(panel[unit]).cumsum()
            panel["_d_cum"] = panel["_treated"].astype(float).groupby(panel[unit]).cumsum()
            panel["_unit_min_t"] = panel.groupby(unit)[time].transform("min")

        # Premean ("max") baseline = mean of all AVAILABLE strictly-prior outcomes.
        # It must NOT depend on the base row's own outcome y_t: PMD replaces the
        # t-1 baseline with the premean of prior periods, and the long difference is
        # y_{t+h} - premean, so a base row with a missing current outcome (but
        # observed priors and target) stays identified. fillna(0) before the
        # cumulative sum makes the numerator the strictly-prior non-missing sum even
        # when y_t (or an interior period) is missing; the denominator counts
        # strictly-prior non-missing outcomes (not rows). Bit-identical to a plain
        # cumsum when no outcome is NaN.
        _filled_outcome = panel[outcome].fillna(0.0)
        outcome_history_sum = _filled_outcome.groupby(panel[unit]).cumsum() - _filled_outcome
        prior_nonnull = panel[outcome].notna().astype(int)
        history_count = prior_nonnull.groupby(panel[unit]).cumsum() - prior_nonnull
        panel["_pmd_all_baseline"] = outcome_history_sum / history_count.replace(0, np.nan)

        lagged_outcome = panel.groupby(unit)[outcome].shift(1)
        if isinstance(self.pmd, int):
            panel["_pmd_k_baseline"] = (
                lagged_outcome.groupby(panel[unit])
                .rolling(window=self.pmd, min_periods=self.pmd)
                .mean()
                .reset_index(level=0, drop=True)
            )

        for lag in range(1, ylags + 1):
            panel[f"_y_lag_{lag}"] = panel.groupby(unit)[outcome].shift(lag)

        panel["_dy_current"] = panel.groupby(unit)[outcome].diff()
        for lag in range(1, dylags + 1):
            panel[f"_dy_lag_{lag}"] = panel.groupby(unit)["_dy_current"].shift(lag)

        if has_gap:
            # Drop the synthetic gap rows. The features above are now calendar-correct
            # on the observed rows (a lag/difference spanning a gap is NaN, so the
            # observation fails closed via the downstream dropna), and no NaN-outcome /
            # NaN-cluster phantom row reaches estimation or the reweight denominators.
            panel = (
                panel.loc[panel["_observed"]]
                .drop(columns="_observed")
                .sort_values([unit, time])
                .reset_index(drop=True)
            )
        return panel

    def _baseline_column(self):
        if self.pmd == "max":
            return "_pmd_all_baseline"
        if isinstance(self.pmd, int):
            return "_pmd_k_baseline"
        return "_baseline_outcome"

    def _clean_control_mask(self, panel: pd.DataFrame, *, time: str, horizon: int) -> pd.Series:
        if self.control_group == "never_treated":
            return panel["_treated"].eq(0) & np.isinf(panel["_first_treat"])

        control_mask = panel["_treated"].eq(0) & panel[time].lt(panel["_first_treat"])
        if horizon >= 0:
            control_mask &= (panel[time] + horizon).lt(panel["_first_treat"])
        return control_mask

    def _nonabsorbing_masks(self, sample, panel, *, unit, time, horizon):
        """Mode-aware (treated_event, clean_control) masks for non-absorbing treatment.

        Dube et al. (2025) Section 4.2 / online Appendix C. Window conditions over
        ``[t-L, t+h]`` are evaluated with offset key-lookups against ``panel``'s
        cumulative columns (``_switch_cum`` = running count of treatment CHANGES, for
        "no change in window"; ``_d_cum`` = running SUM of D, for "D=0 across window").
        Lookups clamp to 0 for periods before a unit's first observed period (boundary
        convention; gap-free panels are enforced in ``_prepare_panel`` so every in-range
        period is present). Returns boolean Series indexed like ``sample``.
        """
        u = sample[unit].to_numpy()
        t = sample[time].to_numpy()
        min_t = sample["_unit_min_t"].to_numpy()
        th = t + horizon

        switch_by_key = panel.set_index([unit, time])["_switch_cum"]
        d_by_key = panel.set_index([unit, time])["_d_cum"]

        def _at(series, times):
            keys = pd.MultiIndex.from_arrays([u, np.asarray(times)])
            vals = series.reindex(keys).to_numpy(dtype=float)
            return np.where(np.asarray(times) < min_t, 0.0, vals)

        if self.non_absorbing == "first_entry":
            # Eq. 12: treated = first entry that stays treated through t+h; control =
            # untreated from start through t+h (== absorbing clean control, reused).
            treated = sample["_entry"].to_numpy(dtype=float) == 1.0
            if horizon >= 0:
                # no exit in (t, t+h]
                treated = treated & (_at(switch_by_key, th) - _at(switch_by_key, t) == 0.0)
            control = self._clean_control_mask(sample, time=time, horizon=horizon).to_numpy()
        else:
            # Eq. 13 (effect stabilization, window L).
            L = self.stabilization_window
            delta_d = sample["_delta_d"].to_numpy(dtype=float)
            if horizon >= 0:
                # treated = fresh entry untreated in [t-L, t-1] (level sum 0) with no other
                # change in (t, t+h]; control = no treatment change in [t-L, t+h] (admits
                # stabilized already-treated units as controls).
                clean_pre = (_at(d_by_key, t - 1) - _at(d_by_key, t - L - 1)) == 0.0
                treated = (delta_d == 1.0) & clean_pre
                treated = treated & (_at(switch_by_key, th) - _at(switch_by_key, t) == 0.0)
                control = (_at(switch_by_key, th) - _at(switch_by_key, t - L - 1)) == 0.0
            else:
                # Placebo horizons: the long-difference y_{t+h} - y_{t-1} reaches back to
                # the target t+h, so the clean window must cover the whole pre-span widened
                # to the stabilization length: [t-M, t-1] with M = max(L, -h). Treated must
                # be untreated across it (a fresh entry contaminated at an earlier period is
                # excluded); controls must have no treatment change across it.
                m = max(L, -horizon)
                clean_pre = (_at(d_by_key, t - 1) - _at(d_by_key, t - m - 1)) == 0.0
                treated = (delta_d == 1.0) & clean_pre
                control = (_at(switch_by_key, t - 1) - _at(switch_by_key, t - m - 1)) == 0.0

        treated_mask = pd.Series(np.asarray(treated, dtype=bool), index=sample.index)
        control_mask = pd.Series(np.asarray(control, dtype=bool), index=sample.index)
        # Treated events have a change at t, clean controls have none -> disjoint by
        # construction; enforce defensively so a row is never classified as both.
        control_mask = control_mask & ~treated_mask
        return treated_mask, control_mask

    def _common_clean_sample_indicator(
        self, panel: pd.DataFrame, *, unit: str, time: str, outcome: str, max_post_horizon: int
    ) -> pd.Series:
        if self.non_absorbing is None:
            common_sample = panel["_entry"].eq(1.0) | self._clean_control_mask(
                panel,
                time=time,
                horizon=max_post_horizon,
            )
        else:
            _treated_mask, _control_mask = self._nonabsorbing_masks(
                panel, panel, unit=unit, time=time, horizon=max_post_horizon
            )
            common_sample = _treated_mask | _control_mask
        # Fixed composition requires the active baseline AND every post-treatment
        # target outcome (h = 0..max_post_horizon) to be NON-MISSING for each base
        # observation -- not merely that the target row exists. This keeps the
        # realized post sample fixed across all post horizons under any missingness
        # encoding (absent rows OR present-but-NaN outcomes).
        outcome_by_key = panel.set_index([unit, time])[outcome]

        def _value_available(horizon: int) -> pd.Series:
            keys = pd.MultiIndex.from_arrays(
                [panel[unit].to_numpy(), (panel[time] + horizon).to_numpy()]
            )
            return pd.Series(outcome_by_key.reindex(keys).notna().to_numpy(), index=panel.index)

        if self.pmd is None:
            available = _value_available(-1)  # t-1 baseline outcome
        else:
            available = panel[self._baseline_column()].notna()
        for h in range(0, max_post_horizon + 1):
            available &= _value_available(h)
        return common_sample & available

    def _rw_weights_from_sample(self, sample: pd.DataFrame) -> pd.Series:
        """Equal-weighting weights from the REALIZED estimation sample.

        Computed after all row drops and clean-control restrictions, so the
        per-event-time denominator matches the regression's actual risk set.
        Computing from the pre-drop panel would silently change the estimand
        (and break the Callaway-Sant'Anna equivalence) on unbalanced panels.
        For each event time, weight = N_clean_control_sample / N_control.
        """
        if sample.empty:
            return pd.Series(dtype=float)
        group_stats = sample.groupby("_event_time")["_entry"].agg(["sum", "count"])
        treated_counts = group_stats["sum"]
        control_counts = group_stats["count"] - treated_counts

        valid = (treated_counts > 0) & (control_counts > 0)
        if not valid.any():
            return pd.Series(dtype=float)

        return (group_stats.loc[valid, "count"] / control_counts.loc[valid]).astype(float)

    def _build_horizon_sample(
        self,
        panel,
        *,
        outcome,
        unit,
        time,
        horizon,
        covariates=None,
        ylags=0,
        dylags=0,
        absorb=None,
        apply_no_composition: bool = True,
    ):
        rhs_columns = self._rhs_column_names(covariates=covariates, ylags=ylags, dylags=dylags)
        base_columns = [
            unit,
            time,
            "_treated",
            "_entry",
            "_first_treat",
            "_cluster",
            "_common_event_ok",
            *(["_delta_d", "_unit_min_t"] if self.non_absorbing is not None else []),
            *rhs_columns,
            *(absorb or []),
        ]
        if self.pmd == "max":
            base_columns.append("_pmd_all_baseline")
        elif isinstance(self.pmd, int):
            base_columns.append("_pmd_k_baseline")

        base = panel[base_columns].copy()
        base["_baseline_time"] = base[time] - 1
        base["_target_time"] = base[time] + horizon

        outcomes = panel[[unit, time, outcome]].copy()

        baseline = outcomes.rename(columns={time: "_baseline_time", outcome: "_baseline_outcome"})
        target = outcomes.rename(columns={time: "_target_time", outcome: "_target_outcome"})

        sample = base.merge(baseline, on=[unit, "_baseline_time"], how="left")
        sample = sample.merge(target, on=[unit, "_target_time"], how="left")
        baseline_column = self._baseline_column()
        # Require the ACTIVE baseline column only: under PMD the long difference
        # uses the premean baseline, so a missing exact t-1 outcome must not drop
        # an otherwise-identified observation (matters on unbalanced panels).
        required_columns = [baseline_column, "_target_outcome", *rhs_columns, *(absorb or [])]
        sample = sample.dropna(subset=required_columns).copy()

        if self.non_absorbing is None:
            treated_mask = sample["_entry"].eq(1.0)
            if self.control_group == "never_treated":
                control_mask = sample["_entry"].eq(0.0) & np.isinf(sample["_first_treat"])
            else:
                control_mask = self._clean_control_mask(sample, time=time, horizon=horizon)
        else:
            treated_mask, control_mask = self._nonabsorbing_masks(
                sample, panel, unit=unit, time=time, horizon=horizon
            )

        sample = sample.loc[treated_mask | control_mask].copy()
        if self.non_absorbing is not None:
            # The per-horizon clean-treated indicator becomes the regression's treatment
            # variable and the treated/control key in every downstream path (estimator,
            # RA split, reweight denominators, identification check). For absorbing and
            # first_entry this equals the original first-entry _entry on the realized
            # sample; under effect_stabilization it also marks re-entry events (which have
            # _entry==0) as treated.
            sample["_entry"] = treated_mask.loc[sample.index].astype(float)
        # Fixed composition is a POST-treatment contract: apply it only to post
        # horizons; pre-treatment placebos use whatever pre-period data exists.
        if self.no_composition and apply_no_composition and horizon >= 0:
            sample = sample.loc[sample["_common_event_ok"]].copy()
        sample["horizon"] = horizon
        sample["_event_time"] = sample[time]
        sample["_long_diff"] = sample["_target_outcome"] - sample[baseline_column]
        return sample[
            [
                "horizon",
                "_event_time",
                "_long_diff",
                "_entry",
                "_cluster",
                *rhs_columns,
                *(absorb or []),
            ]
        ]

    def _sample_is_identified(self, sample: pd.DataFrame) -> bool:
        return len(sample) > 0 and sample["_entry"].nunique() >= 2

    def _build_feature_frame(
        self,
        sample: pd.DataFrame,
        *,
        rhs_columns=None,
        absorb_columns=None,
        include_time_fe: bool = True,
        time_levels=None,
        absorb_levels=None,
    ) -> pd.DataFrame:
        rhs_columns = list(rhs_columns or [])
        absorb_columns = list(absorb_columns or [])
        feature_blocks = []

        for col in rhs_columns:
            values = pd.to_numeric(sample[col], errors="coerce")
            if values.isna().any():
                raise ValueError(
                    f"LPDiD requires numeric covariate-style columns, got invalid values in '{col}'"
                )
            feature_blocks.append(
                pd.DataFrame({col: values.to_numpy(dtype=float)}, index=sample.index)
            )

        if include_time_fe:
            time_categories = list(
                time_levels if time_levels is not None else pd.unique(sample["_event_time"])
            )
            time_dummies = pd.get_dummies(
                pd.Categorical(sample["_event_time"], categories=time_categories),
                prefix="time",
                drop_first=True,
                dtype=float,
            )
            if not time_dummies.empty:
                time_dummies.index = sample.index
                feature_blocks.append(time_dummies)

        for col in absorb_columns:
            categories = list(
                absorb_levels[col] if absorb_levels is not None else pd.unique(sample[col])
            )
            dummies = pd.get_dummies(
                pd.Categorical(sample[col], categories=categories),
                prefix=col,
                drop_first=True,
                dtype=float,
            )
            if not dummies.empty:
                dummies.index = sample.index
                feature_blocks.append(dummies)

        if not feature_blocks:
            return pd.DataFrame(index=sample.index)
        return pd.concat(feature_blocks, axis=1)

    def _estimate_regression_adjustment_sample(
        self,
        sample: pd.DataFrame,
        *,
        response_column: str = "_long_diff",
        include_time_fe: bool = True,
        rhs_columns=None,
        absorb_columns=None,
    ) -> Dict[str, Optional[float]]:
        rhs_columns = list(rhs_columns or [])
        absorb_columns = list(absorb_columns or [])
        dropna_columns = [*rhs_columns, *absorb_columns]
        if dropna_columns:
            sample = sample.dropna(subset=dropna_columns).copy()

        n_obs = int(len(sample))
        empty_result = {
            "coefficient": np.nan,
            "se": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "conf_low": np.nan,
            "conf_high": np.nan,
            "n_obs": n_obs,
            "n_clusters": np.nan,
        }
        if n_obs == 0 or sample["_entry"].nunique() < 2:
            return empty_result

        controls = sample.loc[sample["_entry"].eq(0.0)].copy()
        treated = sample.loc[sample["_entry"].eq(1.0)].copy()
        if controls.empty or treated.empty:
            return empty_result

        # The RA counterfactual for a treated observation is the predicted long
        # difference from the clean-control regression at that event time. An
        # event time with treated units but no clean control has an unidentified
        # time fixed effect, so those treated observations cannot be imputed:
        # drop them (and surface the drop) rather than extrapolate off a
        # rank-deficient fit.
        if include_time_fe:
            control_event_times = set(controls["_event_time"].unique())
            identified = treated["_event_time"].isin(control_event_times)
            if not bool(identified.all()):
                n_drop = int((~identified).sum())
                warnings.warn(
                    f"LPDiD regression adjustment: dropped {n_drop} treated observation(s) "
                    "at event time(s) with no clean control (counterfactual unidentified).",
                    UserWarning,
                    stacklevel=2,
                )
                treated = treated.loc[identified].copy()
                if treated.empty:
                    return empty_result
                sample = pd.concat([controls, treated])
                n_obs = int(len(sample))

        # Absorbed-factor overlap: a treated observation whose absorbed level is
        # absent from the clean controls has an all-zero control dummy for that
        # level, so its counterfactual would be extrapolated through an unidentified
        # coefficient. Drop those treated observations (and surface the drop) rather
        # than impute off a non-identified fit. Mirrors the event-time check above.
        if absorb_columns:
            unsupported = pd.Series(False, index=treated.index)
            for col in absorb_columns:
                control_levels = set(controls[col].unique())
                unsupported = unsupported | ~treated[col].isin(control_levels)
            if bool(unsupported.any()):
                n_drop = int(unsupported.sum())
                warnings.warn(
                    f"LPDiD regression adjustment: dropped {n_drop} treated observation(s) "
                    "with an absorbed-factor level absent from the clean controls "
                    "(counterfactual unidentified -- no overlap).",
                    UserWarning,
                    stacklevel=2,
                )
                treated = treated.loc[~unsupported].copy()
                if treated.empty:
                    return empty_result
                sample = pd.concat([controls, treated])
                n_obs = int(len(sample))

        time_levels = list(pd.unique(sample["_event_time"])) if include_time_fe else None
        absorb_levels = {col: list(pd.unique(sample[col])) for col in absorb_columns}
        control_features = self._build_feature_frame(
            controls,
            rhs_columns=rhs_columns,
            absorb_columns=absorb_columns,
            include_time_fe=include_time_fe,
            time_levels=time_levels,
            absorb_levels=absorb_levels,
        )
        treated_features = self._build_feature_frame(
            treated,
            rhs_columns=rhs_columns,
            absorb_columns=absorb_columns,
            include_time_fe=include_time_fe,
            time_levels=time_levels,
            absorb_levels=absorb_levels,
        )

        control_design = np.column_stack(
            [np.ones(len(controls), dtype=float), control_features.to_numpy(dtype=float)]
        )
        column_names = ["intercept", *control_features.columns.tolist()]
        control_coef, _, _ = solve_ols(
            control_design,
            controls[response_column].to_numpy(dtype=float),
            return_vcov=False,
            rank_deficient_action=self.rank_deficient_action,
            column_names=column_names,
        )
        # solve_ols sets DROPPED redundant-nuisance coefficients to NaN under
        # rank_deficient_action="warn"/"silent" (a constant/duplicate covariate, a
        # collinear absorbed level, or lag collinearity). The dropped column's
        # contribution is absorbed by the retained collinear column(s), so it acts
        # as 0 for prediction/residuals; without this zero-fill the NaN would
        # propagate through every prediction and NaN an otherwise-identified ATT.
        # ("error" still raises inside solve_ols before returning.)
        control_coef = np.where(np.isfinite(control_coef), control_coef, 0.0)

        treated_design = np.column_stack(
            [np.ones(len(treated), dtype=float), treated_features.to_numpy(dtype=float)]
        )
        untreated_prediction = treated_design @ control_coef
        treated_residual = treated[response_column].to_numpy(dtype=float) - untreated_prediction

        effect = float(treated_residual.mean())
        se = np.nan
        cluster_ids = sample["_cluster"].to_numpy()
        n_clusters = len(pd.unique(cluster_ids))
        if n_clusters >= 2 and np.isfinite(effect):
            n_total = len(sample)
            n_treated = len(treated)
            q0_inv, _, _ = _rank_guarded_inv(control_design.T @ control_design)
            mu_treated = treated_design.mean(axis=0)
            control_projection = control_design @ (q0_inv @ mu_treated)

            phi = np.zeros(n_total, dtype=float)
            treated_mask = sample["_entry"].to_numpy(dtype=float) == 1.0
            phi[treated_mask] = (n_total / n_treated) * (treated_residual - effect)
            control_residual = (
                controls[response_column].to_numpy(dtype=float) - control_design @ control_coef
            )
            phi[~treated_mask] = -n_total * control_projection * control_residual

            cluster_scores = pd.Series(phi).groupby(cluster_ids).sum().to_numpy(dtype=float)
            vcov_scalar = float(cluster_scores @ cluster_scores) / float(n_total**2)
            if np.isfinite(vcov_scalar) and vcov_scalar >= 0:
                se = float(np.sqrt(vcov_scalar))

        df = n_clusters - 1 if n_clusters > 1 else None
        t_stat, p_value, conf_int = safe_inference(effect, se, alpha=self.alpha, df=df)
        return {
            "coefficient": effect,
            "se": se,
            "t_stat": t_stat,
            "p_value": p_value,
            "conf_low": conf_int[0],
            "conf_high": conf_int[1],
            "n_obs": n_obs,
            "n_clusters": n_clusters,
        }

    def _estimate_sample(
        self,
        sample: pd.DataFrame,
        *,
        response_column: str = "_long_diff",
        include_time_fe: bool = True,
        rhs_columns=None,
        absorb_columns=None,
        weight_column: Optional[str] = None,
    ) -> Dict[str, Optional[float]]:
        rhs_columns = list(rhs_columns or [])
        absorb_columns = list(absorb_columns or [])
        dropna_columns = [*rhs_columns, *absorb_columns]
        if weight_column is not None:
            dropna_columns.append(weight_column)
        if dropna_columns:
            sample = sample.dropna(subset=dropna_columns).copy()

        n_obs = int(len(sample))
        empty_result = {
            "coefficient": np.nan,
            "se": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "conf_low": np.nan,
            "conf_high": np.nan,
            "n_obs": n_obs,
            "n_clusters": np.nan,
        }

        if n_obs == 0 or sample["_entry"].nunique() < 2:
            return empty_result

        if include_time_fe:
            # Clean-control support: a treated observation at an event time with no
            # clean control has a time fixed effect collinear with the treatment
            # indicator. The rank handler could then drop that time dummy and
            # identify the treatment effect off invalid cross-event-time
            # comparisons, so drop those unsupported treated observations (and
            # surface the drop) rather than emit a spurious estimate. Mirrors the
            # regression-adjustment path's event-time identification check.
            control_event_times = set(sample.loc[sample["_entry"].eq(0.0), "_event_time"].unique())
            unsupported = sample["_entry"].eq(1.0) & ~sample["_event_time"].isin(
                control_event_times
            )
            if bool(unsupported.any()):
                n_drop = int(unsupported.sum())
                warnings.warn(
                    f"LPDiD: dropped {n_drop} treated observation(s) at event time(s) with no "
                    "clean control (the treatment effect is unidentified at that event time).",
                    UserWarning,
                    stacklevel=2,
                )
                sample = sample.loc[~unsupported].copy()
                n_obs = int(len(sample))
                if n_obs == 0 or sample["_entry"].nunique() < 2:
                    return {**empty_result, "n_obs": n_obs}

        design_columns = [
            np.ones(n_obs, dtype=float),
            sample["_entry"].to_numpy(dtype=float),
        ]
        column_names = ["intercept", "treatment_entry"]

        for col in rhs_columns:
            values = pd.to_numeric(sample[col], errors="coerce")
            if values.isna().any():
                raise ValueError(
                    f"LPDiD requires numeric covariate-style columns, got invalid values in '{col}'"
                )
            design_columns.append(values.to_numpy(dtype=float))
            column_names.append(col)

        if include_time_fe:
            time_dummies = pd.get_dummies(
                sample["_event_time"],
                prefix="time",
                drop_first=True,
                dtype=float,
            )
            if not time_dummies.empty:
                design_columns.append(time_dummies.to_numpy(dtype=float))
                column_names.extend(time_dummies.columns.tolist())

        for col in absorb_columns:
            dummies = pd.get_dummies(sample[col], prefix=col, drop_first=True, dtype=float)
            if not dummies.empty:
                design_columns.append(dummies.to_numpy(dtype=float))
                column_names.extend(dummies.columns.tolist())

        design = np.column_stack(design_columns)
        response = sample[response_column].to_numpy(dtype=float)
        cluster_ids = sample["_cluster"].to_numpy()
        weights = None if weight_column is None else sample[weight_column].to_numpy(dtype=float)
        if n_obs <= design.shape[1]:
            coef, _, _ = solve_ols(
                design,
                response,
                return_vcov=False,
                rank_deficient_action=self.rank_deficient_action,
                column_names=column_names,
                weights=weights,
            )
            return {
                "coefficient": float(coef[1]),
                "se": np.nan,
                "t_stat": np.nan,
                "p_value": np.nan,
                "conf_low": np.nan,
                "conf_high": np.nan,
                "n_obs": n_obs,
                "n_clusters": len(pd.unique(cluster_ids)),
            }

        use_cluster_vcov = len(pd.unique(cluster_ids)) >= 2
        vcov = None
        if use_cluster_vcov:
            try:
                coef, _, vcov = solve_ols(
                    design,
                    response,
                    cluster_ids=cluster_ids,
                    return_vcov=True,
                    rank_deficient_action=self.rank_deficient_action,
                    column_names=column_names,
                    weights=weights,
                )
            except (ValueError, ZeroDivisionError):
                coef, _, _ = solve_ols(
                    design,
                    response,
                    return_vcov=False,
                    rank_deficient_action=self.rank_deficient_action,
                    column_names=column_names,
                    weights=weights,
                )
        else:
            coef, _, _ = solve_ols(
                design,
                response,
                return_vcov=False,
                rank_deficient_action=self.rank_deficient_action,
                column_names=column_names,
                weights=weights,
            )

        effect = float(coef[1])
        se = np.nan
        if vcov is not None and vcov.shape[0] > 1 and np.isfinite(vcov[1, 1]) and vcov[1, 1] >= 0:
            se = float(np.sqrt(vcov[1, 1]))

        n_clusters = len(pd.unique(cluster_ids))
        df = n_clusters - 1 if vcov is not None and n_clusters > 1 else None
        t_stat, p_value, conf_int = safe_inference(effect, se, alpha=self.alpha, df=df)
        return {
            "coefficient": effect,
            "se": se,
            "t_stat": t_stat,
            "p_value": p_value,
            "conf_low": conf_int[0],
            "conf_high": conf_int[1],
            "n_obs": n_obs,
            "n_clusters": n_clusters,
        }

    def _estimate_horizon(
        self,
        panel,
        *,
        outcome,
        unit,
        time,
        horizon,
        covariates=None,
        ylags=0,
        dylags=0,
        absorb=None,
    ):
        rhs_columns = self._rhs_column_names(covariates=covariates, ylags=ylags, dylags=dylags)
        sample = self._build_horizon_sample(
            panel,
            outcome=outcome,
            unit=unit,
            time=time,
            horizon=horizon,
            covariates=covariates,
            ylags=ylags,
            dylags=dylags,
            absorb=absorb,
        )
        ra_required = self.reweight and bool(rhs_columns or absorb)
        if ra_required:
            return self._estimate_regression_adjustment_sample(
                sample,
                rhs_columns=rhs_columns,
                absorb_columns=absorb,
            )
        if self.reweight:
            weight_map = self._rw_weights_from_sample(sample)
            sample["_rw_event_weight"] = sample["_event_time"].map(weight_map)
        return self._estimate_sample(
            sample,
            rhs_columns=rhs_columns,
            absorb_columns=absorb,
            weight_column="_rw_event_weight" if self.reweight else None,
        )

    def _build_pooled_sample(
        self,
        panel,
        *,
        outcome,
        unit,
        time,
        horizons,
        kind: str,
        covariates=None,
        ylags=0,
        dylags=0,
        absorb=None,
    ):
        rhs_columns = self._rhs_column_names(covariates=covariates, ylags=ylags, dylags=dylags)
        base_columns = [
            unit,
            time,
            "_treated",
            "_entry",
            "_first_treat",
            "_cluster",
            "_common_pooled_ok",
            *(["_delta_d", "_unit_min_t"] if self.non_absorbing is not None else []),
            *rhs_columns,
            *(absorb or []),
        ]
        if self.pmd == "max":
            base_columns.append("_pmd_all_baseline")
        elif isinstance(self.pmd, int):
            base_columns.append("_pmd_k_baseline")

        base = panel[base_columns].copy()
        base["_baseline_time"] = base[time] - 1

        outcomes = panel[[unit, time, outcome]].copy()
        baseline = outcomes.rename(columns={time: "_baseline_time", outcome: "_baseline_outcome"})
        sample = base.merge(baseline, on=[unit, "_baseline_time"], how="left")

        target_columns = []
        for idx, horizon in enumerate(horizons):
            target_time_col = f"_target_time_{idx}"
            target_outcome_col = f"_target_outcome_{idx}"
            sample[target_time_col] = sample[time] + horizon
            target = outcomes.rename(columns={time: target_time_col, outcome: target_outcome_col})
            sample = sample.merge(target, on=[unit, target_time_col], how="left")
            target_columns.append(target_outcome_col)

        baseline_column = self._baseline_column()
        # Require the ACTIVE baseline column only (see _build_horizon_sample).
        required_columns = [baseline_column, *target_columns, *rhs_columns, *(absorb or [])]
        sample = sample.dropna(subset=required_columns).copy()

        if self.non_absorbing is None:
            treated_mask = sample["_entry"].eq(1.0)
            if self.control_group == "never_treated":
                control_mask = sample["_entry"].eq(0.0) & np.isinf(sample["_first_treat"])
            else:
                control_horizon = max(horizons) if kind == "post" else 0
                control_mask = self._clean_control_mask(sample, time=time, horizon=control_horizon)
        else:
            # Pre pooling reaches back to the MOST NEGATIVE horizon, so the
            # non-absorbing clean window must cover it (the effect_stabilization
            # placebo window widens to [t - max(L, -h), t-1]); using horizon=0 would
            # only clean [t-L, t] and leak prior treatment changes inside the pooled
            # pre reach-back. (Absorbing pre uses horizon=0 above: not-yet-treated at t
            # implies clean across the whole pre-span, so it is unaffected.)
            control_horizon = max(horizons) if kind == "post" else min(horizons)
            treated_mask, control_mask = self._nonabsorbing_masks(
                sample, panel, unit=unit, time=time, horizon=control_horizon
            )

        sample = sample.loc[treated_mask | control_mask].copy()
        if self.non_absorbing is not None:
            sample["_entry"] = treated_mask.loc[sample.index].astype(float)
        if self.no_composition and kind == "post":
            sample = sample.loc[sample["_common_pooled_ok"]].copy()
        sample["_event_time"] = sample[time]
        sample["_pooled_diff"] = sample[target_columns].mean(axis=1) - sample[baseline_column]
        return sample[
            ["_event_time", "_pooled_diff", "_entry", "_cluster", *rhs_columns, *(absorb or [])]
        ]

    def _estimate_window(
        self,
        panel,
        *,
        outcome,
        unit,
        time,
        horizons: Iterable[int],
        kind: str,
        covariates=None,
        ylags=0,
        dylags=0,
        absorb=None,
    ):
        horizons = list(horizons)
        if not horizons:
            raise ValueError(
                f"pooled {kind} window is empty (no horizons to pool); a pooled pre "
                "window requires pre_window >= 2. Use only_event=True or widen the window."
            )
        unidentified_horizons = [
            horizon
            for horizon in horizons
            if not self._sample_is_identified(
                self._build_horizon_sample(
                    panel,
                    outcome=outcome,
                    unit=unit,
                    time=time,
                    horizon=horizon,
                    covariates=covariates,
                    ylags=ylags,
                    dylags=dylags,
                    absorb=absorb,
                    # Base identification only; pooled estimation applies its own
                    # (pooled-window) composition mask via _common_pooled_ok.
                    apply_no_composition=False,
                )
            )
        ]
        if unidentified_horizons:
            raise ValueError(f"unidentified pooled {kind} horizons: {unidentified_horizons}")

        rhs_columns = self._rhs_column_names(covariates=covariates, ylags=ylags, dylags=dylags)
        pooled_sample = self._build_pooled_sample(
            panel,
            outcome=outcome,
            unit=unit,
            time=time,
            horizons=horizons,
            kind=kind,
            covariates=covariates,
            ylags=ylags,
            dylags=dylags,
            absorb=absorb,
        )
        if pooled_sample.empty:
            raise ValueError(f"pooled {kind} window did not contain any horizons")
        ra_required = self.reweight and bool(rhs_columns or absorb)
        if ra_required:
            return self._estimate_regression_adjustment_sample(
                pooled_sample,
                response_column="_pooled_diff",
                include_time_fe=True,
                rhs_columns=rhs_columns,
                absorb_columns=absorb,
            )
        if self.reweight:
            weight_map = self._rw_weights_from_sample(pooled_sample)
            pooled_sample["_rw_pooled_weight"] = pooled_sample["_event_time"].map(weight_map)
        return self._estimate_sample(
            pooled_sample,
            response_column="_pooled_diff",
            include_time_fe=True,
            rhs_columns=rhs_columns,
            absorb_columns=absorb,
            weight_column="_rw_pooled_weight" if self.reweight else None,
        )

    def _resolve_pooled_horizons(self, pooled, *, kind):
        if isinstance(pooled, bool):
            raise ValueError(f"{kind}_pooled must be None, an int, or a length-2 tuple (not bool)")
        if kind == "pre":
            default = list(range(-self.pre_window, -1))
            if isinstance(pooled, int):
                horizons = list(range(-pooled, -1))
            elif pooled is None:
                horizons = default
        else:
            default = list(range(0, self.post_window + 1))
            if isinstance(pooled, int):
                horizons = list(range(0, pooled + 1))
            elif pooled is None:
                horizons = default

        if isinstance(pooled, tuple) and len(pooled) == 2:
            start, end = pooled
            horizons = list(range(start, end + 1))
        elif not (pooled is None or isinstance(pooled, int)):
            raise ValueError(f"{kind}_pooled must be None, an int, or a length-2 tuple")

        if kind == "pre":
            # Exclude h=-1: it is the fixed base-period reference (coefficient 0),
            # not an estimable placebo, so it cannot enter a pooled pre window.
            supported_horizons = set(range(-self.pre_window, -1))
        else:
            supported_horizons = set(range(0, self.post_window + 1))

        invalid_horizons = [horizon for horizon in horizons if horizon not in supported_horizons]
        if invalid_horizons:
            raise ValueError(
                f"Requested pooled {kind} horizons {invalid_horizons} fall outside the supported {kind} "
                f"window {sorted(supported_horizons)}"
            )

        return horizons

    def fit(
        self,
        data,
        outcome,
        unit,
        time,
        treatment,
        covariates=None,
        ylags=0,
        dylags=0,
        absorb=None,
        post_pooled=None,
        pre_pooled=None,
        only_event=False,
        only_pooled=False,
    ):
        self.results_ = None
        self.is_fitted_ = False
        self._fit_meta = None

        # Validate covariate/absorb shape and reserved names BEFORE building the
        # required-columns list, so a string (e.g. absorb="region") raises the
        # precise "must be a list" error rather than an iterate-characters
        # missing-column error, and a name colliding with an LPDiD working column
        # is rejected up front.
        for _arg_name, _arg in (("covariates", covariates), ("absorb", absorb)):
            if isinstance(_arg, str):
                raise ValueError(f"{_arg_name} must be a list of column names, not a string")
            for _col in _arg or []:
                if isinstance(_col, str) and (_col.startswith("_") or _col == "horizon"):
                    raise ValueError(
                        f"{_arg_name} column '{_col}' collides with an LPDiD reserved/internal "
                        "name (names starting with '_' or named 'horizon' are reserved by the "
                        "estimator's working columns); rename it."
                    )

        required = [outcome, unit, time, treatment]
        if covariates:
            required.extend(covariates)
        if absorb:
            required.extend(absorb)
        if self.cluster:
            required.append(self.cluster)
        missing = [col for col in required if col not in data.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        if only_event and only_pooled:
            raise ValueError("only_event and only_pooled cannot both be True")
        if not isinstance(ylags, int) or isinstance(ylags, bool) or ylags < 0:
            raise ValueError("ylags must be a non-negative integer")
        if not isinstance(dylags, int) or isinstance(dylags, bool) or dylags < 0:
            raise ValueError("dylags must be a non-negative integer")
        if not pd.api.types.is_numeric_dtype(data[time]):
            raise ValueError(
                "LPDiD requires a numeric `time` column with integer-spaced periods: "
                "long differences use t-1 / t+h arithmetic on the time labels. Map "
                "irregular or datetime periods to consecutive integers first."
            )
        _unique_times = np.sort(pd.unique(data[time]))
        if not np.allclose(_unique_times, np.round(_unique_times)):
            raise ValueError(
                "LPDiD requires integer-valued `time` labels (e.g. 0, 1, 2 or 2000, "
                "2001, ...): the per-unit calendar grid is built with t-1 / t+h integer "
                "arithmetic. Map fractional or datetime periods to consecutive integers first."
            )
        if len(_unique_times) > 1 and not np.allclose(np.diff(_unique_times), 1.0):
            raise ValueError(
                "LPDiD requires integer-spaced `time` periods (consecutive, spaced by 1); "
                "the global time grid is irregular (e.g. 2000, 2002, ...). Remap periods "
                "to consecutive integers first so t-1 / t+h horizons are well defined."
            )
        if (covariates or absorb or ylags or dylags) and not self.reweight:
            warnings.warn(
                "LPDiD: covariate-style controls (covariates, outcome lags `ylags`, "
                "first-difference lags `dylags`, and absorbed factors) are entering the "
                "long-difference regression by direct inclusion (reweight=False). Per "
                "Dube, Girardi, Jorda & Taylor (2025) online Appendix B.2.2, direct "
                "inclusion preserves the non-negative LP-DiD weighting result only under "
                "linear and homogeneous control effects (Assumption 6 / Appendix B.2.1); "
                "otherwise the implied weights are not guaranteed non-negative. For the "
                "recommended regression-adjustment path, set reweight=True "
                "(Appendix B.2.2 / Section 4.1).",
                UserWarning,
                stacklevel=2,
            )

        cluster = self.cluster or unit
        pre_horizons = self._resolve_pooled_horizons(pre_pooled, kind="pre")
        post_horizons = self._resolve_pooled_horizons(post_pooled, kind="post")
        panel = self._prepare_panel(
            data,
            outcome=outcome,
            unit=unit,
            time=time,
            treatment=treatment,
            cluster=cluster,
            covariates=covariates,
            ylags=ylags,
            dylags=dylags,
            absorb=absorb,
        )
        panel["_common_event_ok"] = self._common_clean_sample_indicator(
            panel,
            unit=unit,
            time=time,
            outcome=outcome,
            max_post_horizon=self.post_window,
        )
        panel["_common_pooled_ok"] = self._common_clean_sample_indicator(
            panel,
            unit=unit,
            time=time,
            outcome=outcome,
            max_post_horizon=max(post_horizons) if post_horizons else 0,
        )
        treatment_by_unit = panel.groupby(unit)["_treated"].max()
        event_study = None
        pooled = None

        if not only_pooled:
            event_rows = []
            for horizon in range(-self.pre_window, self.post_window + 1):
                if horizon == -1:
                    estimate = {
                        "coefficient": 0.0,
                        "se": np.nan,
                        "t_stat": np.nan,
                        "p_value": np.nan,
                        "conf_low": np.nan,
                        "conf_high": np.nan,
                        "n_obs": np.nan,
                    }
                else:
                    estimate = self._estimate_horizon(
                        panel,
                        outcome=outcome,
                        unit=unit,
                        time=time,
                        horizon=horizon,
                        covariates=covariates,
                        ylags=ylags,
                        dylags=dylags,
                        absorb=absorb,
                    )
                event_rows.append(
                    {
                        "horizon": horizon,
                        **estimate,
                    }
                )
            event_study = pd.DataFrame(event_rows)

        if not only_event:
            pre_estimate = self._estimate_window(
                panel,
                outcome=outcome,
                unit=unit,
                time=time,
                horizons=pre_horizons,
                kind="pre",
                covariates=covariates,
                ylags=ylags,
                dylags=dylags,
                absorb=absorb,
            )
            post_estimate = self._estimate_window(
                panel,
                outcome=outcome,
                unit=unit,
                time=time,
                horizons=post_horizons,
                kind="post",
                covariates=covariates,
                ylags=ylags,
                dylags=dylags,
                absorb=absorb,
            )
            pooled = pd.DataFrame(
                [
                    {
                        "window": "pre",
                        **pre_estimate,
                    },
                    {
                        "window": "post",
                        **post_estimate,
                    },
                ]
            )

        # Headline G = realized clusters in the pooled-post (headline ATT) sample
        # when computed; otherwise the panel-level unit-cluster count. Per-horizon
        # rows carry their own realized n_clusters in the event_study/pooled tables.
        headline_n_clusters = int(panel["_cluster"].nunique())
        if pooled is not None:
            _post = pooled.loc[pooled["window"] == "post", "n_clusters"]
            if not _post.empty and pd.notna(_post.iloc[0]):
                headline_n_clusters = int(_post.iloc[0])

        self.results_ = LPDiDResults(
            event_study=event_study,
            pooled=pooled,
            n_obs=len(data),
            n_treated_units=int(treatment_by_unit.gt(0).sum()),
            n_control_units=int(treatment_by_unit.eq(0).sum()),
            pre_window=self.pre_window,
            post_window=self.post_window,
            control_group=self.control_group,
            reweight=self.reweight,
            no_composition=self.no_composition,
            pmd=self.pmd,
            alpha=self.alpha,
            cluster_name=cluster,
            n_clusters=headline_n_clusters,
            vcov_type=(
                "if_cluster"
                if (self.reweight and bool(covariates or ylags or dylags or absorb))
                else "hc1"
            ),
            rank_deficient_action=self.rank_deficient_action,
            covariates=list(covariates) if covariates else None,
            absorb=list(absorb) if absorb else None,
            ylags=ylags,
            dylags=dylags,
            non_absorbing=self.non_absorbing,
            stabilization_window=self.stabilization_window,
        )
        self._fit_meta = {"cluster": cluster, "outcome": outcome, "unit": unit, "time": time}
        self.is_fitted_ = True
        return self.results_

    def get_params(self) -> Dict[str, Any]:
        return {
            "pre_window": self.pre_window,
            "post_window": self.post_window,
            "control_group": self.control_group,
            "reweight": self.reweight,
            "no_composition": self.no_composition,
            "pmd": self.pmd,
            "alpha": self.alpha,
            "cluster": self.cluster,
            "rank_deficient_action": self.rank_deficient_action,
            "non_absorbing": self.non_absorbing,
            "stabilization_window": self.stabilization_window,
        }

    def set_params(self, **params: Any) -> "LPDiD":
        previous_values = {}
        for key, value in params.items():
            if hasattr(self, key):
                previous_values[key] = getattr(self, key)
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")
        try:
            self._validate_params()
        except ValueError:
            for key, value in previous_values.items():
                setattr(self, key, value)
            raise
        return self
