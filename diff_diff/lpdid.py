from typing import Any, Dict, Iterable, Optional, Union

import numpy as np
import pandas as pd

from diff_diff.linalg import solve_ols
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
        self._validate_params()
        self.is_fitted_ = False
        self.results_: Optional[LPDiDResults] = None

    def _validate_params(self) -> None:
        if self.control_group not in ("clean", "never_treated"):
            raise ValueError("control_group must be 'clean' or 'never_treated'")
        if self.rank_deficient_action not in ("warn", "error", "silent"):
            raise ValueError("rank_deficient_action must be 'warn', 'error', or 'silent'")
        if self.pmd is not None and not (
            self.pmd == "max" or (isinstance(self.pmd, int) and not isinstance(self.pmd, bool) and self.pmd > 0)
        ):
            raise ValueError("pmd must be None, 'max', or a positive integer")

    def _prepare_panel(self, data, outcome, unit, time, treatment, cluster):
        selected_columns = list(dict.fromkeys([unit, time, outcome, treatment, cluster]))
        panel = data[selected_columns].copy()
        panel = panel.sort_values([unit, time]).reset_index(drop=True)

        if panel.duplicated([unit, time]).any():
            raise ValueError("LPDiD requires unique unit-time observations")

        treated_numeric = pd.to_numeric(panel[treatment], errors="coerce")
        if treated_numeric.isna().any() or not treated_numeric.isin([0, 1]).all():
            raise ValueError("treatment must contain binary numeric 0/1 values with no missing data")

        panel["_treated"] = treated_numeric.astype(int)
        panel["_cluster"] = panel[cluster]
        panel["_lag_treated"] = panel.groupby(unit)["_treated"].shift(1, fill_value=0)
        panel["_entry"] = ((panel["_treated"] == 1) & (panel["_lag_treated"] == 0)).astype(float)
        panel["_treated_cummax"] = panel.groupby(unit)["_treated"].cummax()

        violating_units = panel.loc[panel["_treated_cummax"] > panel["_treated"], unit].unique()
        if len(violating_units) > 0:
            raise ValueError(
                "LPDiD currently requires an absorbing treatment path "
                "(once treated, always treated)"
            )

        first_treat = panel.loc[panel["_entry"] == 1].groupby(unit)[time].min()
        panel["_first_treat"] = panel[unit].map(first_treat).astype(float).fillna(np.inf)
        return panel

    def _build_horizon_sample(self, panel, *, outcome, unit, time, horizon):
        base = panel[[unit, time, "_entry", "_first_treat", "_cluster"]].copy()
        base["_baseline_time"] = base[time] - 1
        base["_target_time"] = base[time] + horizon

        outcomes = panel[[unit, time, outcome]].copy()

        baseline = outcomes.rename(columns={time: "_baseline_time", outcome: "_baseline_outcome"})
        target = outcomes.rename(columns={time: "_target_time", outcome: "_target_outcome"})

        sample = base.merge(baseline, on=[unit, "_baseline_time"], how="left")
        sample = sample.merge(target, on=[unit, "_target_time"], how="left")
        sample = sample.dropna(subset=["_baseline_outcome", "_target_outcome"]).copy()

        treated_mask = sample["_entry"].eq(1.0)
        if self.control_group == "never_treated":
            control_mask = sample["_entry"].eq(0.0) & np.isinf(sample["_first_treat"])
        else:
            control_mask = sample["_entry"].eq(0.0) & sample[time].lt(sample["_first_treat"])
            if horizon >= 0:
                control_mask &= sample["_target_time"].lt(sample["_first_treat"])

        sample = sample.loc[treated_mask | control_mask].copy()
        sample["horizon"] = horizon
        sample["_long_diff"] = sample["_target_outcome"] - sample["_baseline_outcome"]
        return sample[["horizon", "_long_diff", "_entry", "_cluster"]]

    def _sample_is_identified(self, sample: pd.DataFrame) -> bool:
        return len(sample) > 0 and sample["_entry"].nunique() >= 2

    def _estimate_sample(self, sample: pd.DataFrame) -> Dict[str, Optional[float]]:
        n_obs = int(len(sample))
        empty_result = {
            "coefficient": np.nan,
            "se": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "conf_low": np.nan,
            "conf_high": np.nan,
            "n_obs": n_obs,
        }

        if n_obs == 0 or sample["_entry"].nunique() < 2:
            return empty_result

        design_columns = [
            np.ones(n_obs, dtype=float),
            sample["_entry"].to_numpy(dtype=float),
        ]
        column_names = ["intercept", "treatment_entry"]

        if sample["horizon"].nunique() > 1:
            horizon_dummies = pd.get_dummies(sample["horizon"], prefix="horizon", drop_first=True, dtype=float)
            if not horizon_dummies.empty:
                design_columns.append(horizon_dummies.to_numpy(dtype=float))
                column_names.extend(horizon_dummies.columns.tolist())

        design = np.column_stack(design_columns)
        response = sample["_long_diff"].to_numpy(dtype=float)
        cluster_ids = sample["_cluster"].to_numpy()
        if n_obs <= design.shape[1]:
            coef, _, _ = solve_ols(
                design,
                response,
                return_vcov=False,
                rank_deficient_action=self.rank_deficient_action,
            )
            return {
                "coefficient": float(coef[1]),
                "se": np.nan,
                "t_stat": np.nan,
                "p_value": np.nan,
                "conf_low": np.nan,
                "conf_high": np.nan,
                "n_obs": n_obs,
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
                )
            except (ValueError, ZeroDivisionError):
                coef, _, _ = solve_ols(
                    design,
                    response,
                    return_vcov=False,
                    rank_deficient_action=self.rank_deficient_action,
                )
        else:
            coef, _, _ = solve_ols(
                design,
                response,
                return_vcov=False,
                rank_deficient_action=self.rank_deficient_action,
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
        }

    def _estimate_horizon(self, panel, *, outcome, unit, time, horizon):
        sample = self._build_horizon_sample(panel, outcome=outcome, unit=unit, time=time, horizon=horizon)
        return self._estimate_sample(sample)

    def _estimate_window(self, panel, *, outcome, unit, time, horizons: Iterable[int], kind: str):
        samples = []
        unidentified_horizons = []
        for horizon in horizons:
            sample = self._build_horizon_sample(panel, outcome=outcome, unit=unit, time=time, horizon=horizon)
            if self._sample_is_identified(sample):
                samples.append(sample)
            else:
                unidentified_horizons.append(horizon)

        if unidentified_horizons:
            raise ValueError(f"unidentified pooled {kind} horizons: {unidentified_horizons}")

        stacked = pd.concat(samples, ignore_index=True) if samples else pd.DataFrame()
        if stacked.empty:
            raise ValueError(f"pooled {kind} window did not contain any horizons")
        return self._estimate_sample(stacked)

    def _resolve_pooled_horizons(self, pooled, *, kind):
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
            supported_horizons = set(range(-self.pre_window, 0))
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

        cluster = self.cluster or unit
        panel = self._prepare_panel(data, outcome=outcome, unit=unit, time=time, treatment=treatment, cluster=cluster)
        treatment_by_unit = panel.groupby(unit)["_treated"].max()
        event_study = None
        pooled = None

        if not only_pooled:
            event_rows = []
            for horizon in range(-self.pre_window, self.post_window + 1):
                estimate = self._estimate_horizon(panel, outcome=outcome, unit=unit, time=time, horizon=horizon)
                event_rows.append(
                    {
                        "horizon": horizon,
                        **estimate,
                    }
                )
            event_study = pd.DataFrame(event_rows)

        if not only_event:
            pre_horizons = self._resolve_pooled_horizons(pre_pooled, kind="pre")
            post_horizons = self._resolve_pooled_horizons(post_pooled, kind="post")
            pre_estimate = self._estimate_window(
                panel, outcome=outcome, unit=unit, time=time, horizons=pre_horizons, kind="pre"
            )
            post_estimate = self._estimate_window(
                panel, outcome=outcome, unit=unit, time=time, horizons=post_horizons, kind="post"
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
