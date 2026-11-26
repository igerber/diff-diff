"""Chained difference-in-differences on overlapping incomplete panels."""

import warnings
from itertools import combinations
from typing import List, Optional

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype
from scipy.special import expit

from diff_diff._base import BaseEstimator
from diff_diff.aggregation import AggregationKit
from diff_diff.chained_did_results import ChainedDiDResults
from diff_diff.linalg import solve_logit
from diff_diff.staggered_results import GroupTimeEffect
from diff_diff.utils import ClusterArg, safe_inference, validate_cluster

_MIN_PAIR_COUNT = 2
_LOW_PAIR_COUNT = 20


def _pair_ipw(changes, treated, covariates=None):
    """Return a normalised pair contrast and O(1/n) unit contributions."""
    nt = int(treated.sum())
    nc = len(treated) - nt
    if min(nt, nc) < _MIN_PAIR_COUNT:
        raise ValueError("Pair IPW requires at least two observed units in each arm")
    info = {"n_treated": nt, "n_control": nc}
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        mt = changes[treated].mean()
        if covariates is None or not covariates.shape[1]:
            mc = changes[~treated].mean()
            contributions = np.where(treated, (changes - mt) / nt, -(changes - mc) / nc)
            info.update(
                propensity_min=nt / len(treated),
                propensity_max=nt / len(treated),
                control_ess=float(nc),
            )
        else:
            # Reparameterise before forming the score Jacobian to preserve affine scale invariance.
            scale = np.maximum(np.abs(covariates).max(axis=0), np.finfo(float).tiny)
            x = covariates / scale
            x = x - x.mean(axis=0)
            spread = np.sqrt(np.mean(x * x, axis=0))
            if (spread == 0).any():
                raise ValueError("IPW covariates are rank deficient (constant column)")
            x = x / spread
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", UserWarning)
                beta, _ = solve_logit(
                    x, treated.astype(float), rank_deficient_action="error", epv_threshold=0
                )
            if caught:
                raise ValueError(f"IPW propensity fit failed: {caught[0].message}")
            z = np.column_stack([np.ones(len(x)), x])
            p = expit(z @ beta)
            if not np.isfinite(beta).all() or ((p <= 0) | (p >= 1)).any():
                raise ValueError("IPW propensity fit has non-finite or boundary probabilities")
            score_beta = z * (treated - p)[:, None]
            if np.max(np.abs(score_beta.sum(axis=0))) > 1e-8 * len(x):
                raise ValueError("IPW propensity fit did not solve its score equations")
            odds = p / (1 - p)
            weights = (~treated) * odds
            mass = weights.sum()
            mc = weights @ changes / mass
            scores = np.column_stack(
                [score_beta, treated * (changes - mt), weights * (changes - mc)]
            )
            k = z.shape[1]
            jacobian = np.zeros((k + 2, k + 2))
            jacobian[:k, :k] = -(z.T * (p * (1 - p))) @ z
            jacobian[k, k] = -nt
            jacobian[k + 1, :k] = (weights * (changes - mc)) @ z
            jacobian[k + 1, k + 1] = -mass
            try:
                parameter_if = -np.linalg.solve(jacobian, scores.T).T
            except np.linalg.LinAlgError as exc:
                raise ValueError("IPW score Jacobian is singular") from exc
            contributions = parameter_if[:, k] - parameter_if[:, k + 1]
            info.update(
                propensity_min=float(p.min()),
                propensity_max=float(p.max()),
                control_ess=float(mass**2 / (weights @ weights)),
            )
        effect = mt - mc
    if not np.isfinite(effect) or not np.isfinite(contributions).all():
        raise ValueError("Non-finite IPW moments or influence contributions")
    return float(effect), contributions, info


class ChainedDiD(BaseEstimator):
    """Estimate chained ATT(g,t) from overlapping outcome observations.

    Uses normalised IPW comparisons, never-treated controls, PT-post and
    fixed identity-weighted GMM. Earlier trend differences remain free.
    Each used observation pair must preserve the joint distribution of
    covariates and actual outcome changes within its cohort. There is no
    correction for covariate-dependent attrition.

    Parameters
    ----------
    alpha : float, default=0.05
        Significance level for pointwise intervals and tests.
    cluster : str, False or None, default=None
        Unit-constant cluster column for one-way CR1 covariance. Without
        clustering (None or False), inference uses unit-level HC0 covariance.
    Notes
    -----
    Treatment is binary and absorbing with zero anticipation. Treatment
    dates must be positive stored numeric dates; 0 denotes never treated.
    No optimal GMM weighting, survey weights or simultaneous bands are offered.
    Pairs need at least two observed units per arm. Exclusions are recorded;
    singleton exclusions and retained pairs below 20 units per arm or control
    effective sample size 20 trigger a warning. These safeguards do not establish
    interval calibration. Identity GMM ignores pair precision, so sparse pairs
    can reduce precision and yield anti-conservative intervals.
    See Bellégo, Benatia and Dortet-Bernadet, doi:10.1016/j.jeconom.2024.105783.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        cluster: ClusterArg = None,
    ) -> None:
        self.alpha: float = alpha
        self.cluster: ClusterArg = cluster
        self.is_fitted_: bool = False
        self.results_: Optional[ChainedDiDResults] = None
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        if isinstance(self.alpha, (bool, np.bool_)) or not isinstance(
            self.alpha, (int, float, np.integer, np.floating)
        ):
            raise ValueError("alpha must be a finite number between 0 and 1")
        if not (np.isfinite(float(self.alpha)) and 0 < float(self.alpha) < 1):
            raise ValueError("alpha must be a finite number between 0 and 1")
        validate_cluster(self.cluster)

    def fit(
        self,
        data: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        first_treat: str,
        covariates: Optional[List[str]] = None,
    ) -> ChainedDiDResults:
        """Fit all supported pairs without balancing or imputing the panel.

        Parameters
        ----------
        data : pandas.DataFrame
            Long panel. NaN outcomes and absent unit-date rows are unobserved.
            Every supplied unit remains in population accounting, including
            units with zero or one finite outcome. Infinity is invalid.
        outcome, unit, time, first_treat : str
            Column names. Cohort membership is supplied, never inferred from
            missing outcomes. Every treated date requires a preceding date.
        covariates : list of str or None
            Complete static numeric covariates, with an automatically added
            intercept. Omit for unconditional differences in mean changes.

        Returns
        -------
        ChainedDiDResults
            Canonical results with per-cell support and full joint covariance.

        Raises
        ------
        ValueError
            Invalid data, unsupported timing or a failed required nuisance fit.
        """
        self.is_fitted_, self.results_ = False, None
        self._validate_parameters()
        covariates = [] if covariates is None else covariates
        if not isinstance(covariates, list) or any(not isinstance(c, str) for c in covariates):
            raise ValueError("covariates must be a list of column names")
        roles = [outcome, unit, time, first_treat] + covariates
        if any(not isinstance(c, str) for c in roles) or len(set(roles)) != len(roles):
            raise ValueError("Column roles and covariates must have distinct column names")
        if not isinstance(data, pd.DataFrame) or data.empty or not data.columns.is_unique:
            raise ValueError("data must be a non-empty DataFrame with unique columns")
        cluster = self.cluster if isinstance(self.cluster, str) else None
        required = roles + ([cluster] if cluster is not None else [])
        missing = [c for c in required if c not in data.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        static = list(
            dict.fromkeys([first_treat] + covariates + ([cluster] if cluster is not None else []))
        )
        if data[[unit, time] + static].isna().any().any():
            raise ValueError("Unit, time, cohort, covariates and cluster metadata must be complete")
        if data.duplicated([unit, time]).any():
            raise ValueError("Duplicate unit-date rows are not allowed")
        for col in [time, first_treat] + covariates:
            if (
                not is_numeric_dtype(data[col])
                or is_complex_dtype(data[col])
                or is_bool_dtype(data[col])
                or not np.isfinite(data[col].to_numpy(dtype=float)).all()
            ):
                raise ValueError(f"Column {col!r} must contain finite numeric values")
        if not is_numeric_dtype(data[outcome]) or is_complex_dtype(data[outcome]):
            raise ValueError("Outcomes must be numeric, with NaN allowed for missing values")
        outcomes = data[outcome].to_numpy(dtype=float, na_value=np.nan)
        if np.isinf(outcomes).any():
            raise ValueError("Infinite outcomes are not allowed")
        grouped = data.groupby(unit, sort=False, observed=True)
        if (grouped[static].nunique(dropna=False) > 1).any().any():
            raise ValueError("Cohort, covariates and cluster must be constant within each unit")
        registry = grouped[static].first()
        times = sorted(data[time].unique().tolist())
        cohorts = sorted(registry[first_treat].unique().tolist())
        if 0 not in cohorts or len(cohorts) < 2:
            raise ValueError("At least one treated cohort and never-treated controls are required")
        cohorts.remove(0)
        if any(g <= 0 or g not in times or g == times[0] for g in cohorts):
            raise ValueError(
                "Each positive treatment date must be stored and have a preceding date"
            )
        y = (
            data.pivot(index=unit, columns=time, values=outcome)
            .reindex(index=registry.index, columns=times)
            .to_numpy(dtype=float, na_value=np.nan)
        )
        memberships = registry[first_treat].to_numpy()
        x = registry[covariates].to_numpy(dtype=float) if covariates else None
        n = len(registry)
        cells, influence, catalogue, link_info, ranks = {}, {}, [], [], {}
        excluded_links = []
        controls = memberships == 0
        for g in cohorts:
            base = times.index(g) - 1
            columns = [j for j in range(len(times)) if j != base]
            rows, differences, pair_if, pair_lookup = [], [], [], {}
            for s, t in combinations(range(len(times)), 2):
                observed = np.isfinite(y[:, s]) & np.isfinite(y[:, t])
                target = memberships == g
                nt = int((observed & target).sum())
                nc = int((observed & controls).sum())
                if min(nt, nc) < _MIN_PAIR_COUNT:
                    reason = (
                        "no_treated_observations"
                        if nt == 0
                        else (
                            "no_control_observations"
                            if nc == 0
                            else "insufficient_pair_replication"
                        )
                    )
                    excluded_links.append(
                        {
                            "group": g,
                            "s": times[s],
                            "t": times[t],
                            "n_treated": nt,
                            "n_control": nc,
                            "reason": reason,
                        }
                    )
                    continue
                mask = observed & (target | controls)
                with np.errstate(over="ignore", invalid="ignore"):
                    delta = y[mask, t] - y[mask, s]
                if not np.isfinite(delta).all():
                    raise ValueError(
                        f"Non-finite outcome changes for cohort {g}, pair {times[s], times[t]}"
                    )
                estimate, q, info = _pair_ipw(
                    delta, target[mask], x[mask] if x is not None else None
                )
                info["low_support"] = bool(min(nt, nc, info["control_ess"]) < _LOW_PAIR_COUNT)
                row = np.zeros(len(columns))
                if s != base:
                    row[columns.index(s)] = -1
                if t != base:
                    row[columns.index(t)] = 1
                full_q = np.zeros(n)
                full_q[mask] = q
                pair_lookup[s, t] = len(rows)
                rows.append(row)
                differences.append(estimate)
                pair_if.append(full_q)
                catalogue.append((g, times[s], times[t]))
                link_info.append(
                    {"group": g, "s": times[s], "t": times[t], "att": estimate, **info}
                )
            a = np.asarray(rows).reshape(-1, len(columns))
            d = np.asarray(differences)
            q = np.column_stack(pair_if) if pair_if else np.empty((n, 0))
            u, singular, vt = np.linalg.svd(a, full_matrices=False)
            cutoff = max(a.shape) * np.finfo(float).eps * (singular[0] if len(singular) else 0)
            keep = singular > cutoff
            basis = vt[keep]
            inverse = (basis.T / singular[keep]) @ u[:, keep].T
            ranks[g] = int(keep.sum())
            for t, label in enumerate(times):
                estimate, contribution, reason = np.nan, np.full(n, np.nan), "unsupported_target"
                if t == base:
                    estimate, contribution, reason = 0.0, np.zeros(n), None
                elif t < base:
                    k = pair_lookup.get((t, base))
                    if k is not None:
                        estimate, contribution, reason = -d[k], -q[:, k], None
                    else:
                        reason = "unsupported_direct_pre_comparison"
                else:
                    contrast = np.zeros(len(columns))
                    contrast[columns.index(t)] = 1
                    residual = contrast - (contrast @ basis.T) @ basis
                    if np.linalg.norm(residual) <= 4 * max(a.shape) * np.finfo(float).eps:
                        loading = contrast @ inverse
                        estimate, contribution, reason = float(loading @ d), q @ loading, None
                contribution = contribution - contribution.mean()
                cells[g, label] = GroupTimeEffect(
                    g,
                    label,
                    estimate,
                    np.nan,
                    np.nan,
                    np.nan,
                    (np.nan, np.nan),
                    int((memberships == g).sum()),
                    int(controls.sum()),
                    reason,
                )
                influence[g, label] = contribution
        singletons = sum(row["reason"] == "insufficient_pair_replication" for row in excluded_links)
        low_support = sum(row["low_support"] for row in link_info)
        if singletons or low_support:
            warnings.warn(
                f"ChainedDiD pair support: {singletons} singleton pairs excluded; "
                f"{low_support} retained pairs have low support (fewer than 20 units "
                "in either arm or control effective sample size below 20). "
                "Inspect diagnostics['excluded_links'] and diagnostics['links']. "
                "Identity GMM ignores pair precision; pointwise intervals may be "
                "anti-conservative. These thresholds do not establish calibration.",
                UserWarning,
                stacklevel=2,
            )
        kit = AggregationKit(
            bookkeeping={
                "unit_cohorts": memberships.copy(),
                "all_units": registry.index.to_numpy(copy=True),
                "clusters": registry[cluster].to_numpy(copy=True) if cluster is not None else None,
                "times": times,
                "cohorts": cohorts,
                "references": {g: times[times.index(g) - 1] for g in cohorts},
            },
            influence=influence,
            alpha=self.alpha,
            anticipation=0,
            cband=False,
        )
        result = ChainedDiDResults(
            cells,
            kit,
            {
                "catalogue": catalogue,
                "links": link_info,
                "excluded_links": excluded_links,
                "pair_support": {
                    "minimum_per_arm": _MIN_PAIR_COUNT,
                    "warning_threshold": _LOW_PAIR_COUNT,
                },
                "rank": ranks,
                "gmm_weighting": "identity",
                "selection": "none",
            },
        )
        covariance = result._covariance(result.influence_functions)
        result.vcov = covariance
        for k, cell in enumerate(cells.values()):
            cell.se = float(np.sqrt(covariance[k, k]))
            cell.t_stat, cell.p_value, cell.conf_int = safe_inference(
                cell.effect, cell.se, self.alpha
            )
        result.att, result.se, result.t_stat, result.p_value, result.conf_int = result._overall()
        self.results_, self.is_fitted_ = result, True
        return result
