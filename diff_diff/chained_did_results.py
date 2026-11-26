"""Canonical results and retained-IF aggregation for chained DiD."""

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from diff_diff.aggregation import AggregationKit, AggregationMixin, AggregationResult
from diff_diff.results_base import (
    _SUMMARY_ALPHA_MESSAGE,
    BaseResults,
    EventStudyResults,
    _require_fit_alpha,
)
from diff_diff.staggered_results import GroupTimeEffect
from diff_diff.utils import safe_inference


@dataclass
class ChainedDiDResults(BaseResults, AggregationMixin):
    """Chained ATT(g,t), direct earlier diagnostics and analytical inference.

    ``att`` is the cohort-mass-weighted average of all scheduled post-treatment
    cells. An unsupported required cell makes its summary unavailable, without
    changing its population. ``vcov`` is on the estimator scale, indexed by
    ``vcov_index``. ``influence_functions`` contains O(1/n) unit contributions,
    including the zero rows of units without an observed change. A non-zero
    pre-reference diagnostic is not a violation of maintained PT-post.
    The retained aggregation kit contains unit metadata and contributions,
    not the outcome panel or an estimator reference.
    """

    group_time_effects: Dict[Tuple[Any, Any], GroupTimeEffect]
    _aggregation_kit: AggregationKit = field(repr=False)
    diagnostics: Dict[str, Any]
    att: float = np.nan
    se: float = np.nan
    t_stat: float = np.nan
    p_value: float = np.nan
    conf_int: Tuple[float, float] = (np.nan, np.nan)
    vcov: Optional[np.ndarray] = field(default=None, repr=False)

    _AGGREGATE_SUPPORTED: ClassVar[Tuple[str, ...]] = ("simple", "event_study")
    _AGGREGATE_BALANCE_E_TYPES: ClassVar[Tuple[str, ...]] = ()

    @property
    def alpha(self) -> float:
        """Fit-time significance level."""
        return self._aggregation_kit.alpha

    @property
    def n_units(self) -> int:
        """Number of original supplied units, including non-repeat observers."""
        return len(self._aggregation_kit.bookkeeping["unit_cohorts"])

    @property
    def vcov_index(self) -> list:
        """Ordered (cohort, date) keys indexing both covariance axes."""
        return list(self.group_time_effects)

    @property
    def influence_functions(self) -> np.ndarray:
        """O(1/n) unit contributions in vcov_index order."""
        return np.column_stack([self._aggregation_kit.influence[k] for k in self.vcov_index])

    def _covariance(self, contributions):
        q = np.asarray(contributions)
        if q.ndim == 1:
            q = q[:, None]
        clusters = self._aggregation_kit.bookkeeping["clusters"]
        if clusters is None:
            return q.T @ q
        codes, labels = pd.factorize(clusters, sort=False)
        if len(labels) < 2:
            return np.full((q.shape[1], q.shape[1]), np.nan)
        sums = np.zeros((len(labels), q.shape[1]))
        np.add.at(sums, codes, q)
        return len(labels) / (len(labels) - 1) * (sums.T @ sums)

    def _combine(self, keys):
        n = self.n_units
        membership = self._aggregation_kit.bookkeeping["unit_cohorts"]
        if not keys or any(self.group_time_effects[k].skip_reason is not None for k in keys):
            return np.nan, np.full(n, np.nan), np.nan
        indicators = np.column_stack([membership == g for g, _ in keys]).astype(float)
        masses = indicators.sum(axis=0)
        denominator = masses.sum()
        weights = self._cell_weights(keys)
        effects = np.array([self.group_time_effects[k].effect for k in keys])
        q = np.column_stack([self._aggregation_kit.influence[k] for k in keys])
        centred = indicators - masses / n
        weight_if = (centred - centred.sum(axis=1)[:, None] * weights) / denominator
        contribution = q @ weights + weight_if @ effects
        return float(weights @ effects), contribution, float(denominator)

    def _post_keys(self):
        return [(g, t) for g, t in self.group_time_effects if t >= g]

    def _event_keys(self, horizon):
        keys = [(g, t) for g, t in self.group_time_effects if t - g == horizon]
        references = self._aggregation_kit.bookkeeping["references"]
        estimated = [(g, t) for g, t in keys if t != references[g]]
        return estimated or keys

    def _cell_weights(self, keys):
        membership = self._aggregation_kit.bookkeeping["unit_cohorts"]
        masses = np.array([np.count_nonzero(membership == g) for g, _ in keys], dtype=float)
        return masses / masses.sum()

    def aggregation_weights(self, level: str = "simple") -> pd.DataFrame:
        """Return constituent cell weights, including intended unavailable cells.

        ``level`` is ``'simple'`` or ``'event_study'``. Columns are ``label``,
        ``group``, ``time``, ``weight`` and cell ``available``. Unsupported
        cells keep their intended weights; the corresponding summary remains
        unavailable. Mixed event rows exclude structural reference identities.
        The returned table is detached from the fitted results.
        """
        if level == "simple":
            summaries = {"overall": self._post_keys()}
        elif level == "event_study":
            summaries = {
                e: self._event_keys(e) for e in sorted({t - g for g, t in self.group_time_effects})
            }
        else:
            raise ValueError("level must be 'simple' or 'event_study'")
        return pd.DataFrame(
            [
                {
                    "label": label,
                    "group": g,
                    "time": t,
                    "weight": weight,
                    "available": self.group_time_effects[g, t].skip_reason is None,
                }
                for label, keys in summaries.items()
                for (g, t), weight in zip(keys, self._cell_weights(keys))
            ],
            columns=["label", "group", "time", "weight", "available"],
        )

    def _overall(self):
        effect, contribution, _ = self._combine(self._post_keys())
        se = float(np.sqrt(self._covariance(contribution)[0, 0]))
        return (effect, se, *safe_inference(effect, se, self.alpha))

    def _aggregate_compute(self, level, *, weights, balance_e):
        if level == "simple":
            return AggregationResult(
                level="simple",
                label=np.array(["overall"]),
                target=np.array(["att"]),
                att=np.array([self.att]),
                se=np.array([self.se]),
                t_stat=np.array([self.t_stat]),
                p_value=np.array([self.p_value]),
                conf_int_lower=np.array([self.conf_int[0]]),
                conf_int_upper=np.array([self.conf_int[1]]),
                n=np.array([self.n_units]),
                weight=np.array([1.0]),
                df=np.array([np.nan]),
                alpha=self.alpha,
                n_kind="units",
                estimator="ChainedDiD",
            )
        horizons = sorted({t - g for g, t in self.group_time_effects})
        effects, ses, statistics, pvalues, lower, upper, refs, counts, contributions = (
            [] for _ in range(9)
        )
        references = self._aggregation_kit.bookkeeping["references"]
        for e in horizons:
            keys = self._event_keys(e)
            estimate, q, _ = self._combine(keys)
            reference = all(t == references[g] for g, t in keys)
            se = float(np.sqrt(self._covariance(q)[0, 0]))
            statistic, pvalue, interval = safe_inference(estimate, se, self.alpha)
            effects.append(estimate)
            ses.append(se)
            statistics.append(statistic)
            pvalues.append(pvalue)
            lower.append(interval[0])
            upper.append(interval[1])
            refs.append(reference)
            counts.append(sum(self.group_time_effects[k].n_treated for k in keys))
            contributions.append(q)
        return EventStudyResults(
            event_time=np.array(horizons),
            att=np.array(effects),
            se=np.array(ses),
            t_stat=np.array(statistics),
            p_value=np.array(pvalues),
            conf_int_lower=np.array(lower),
            conf_int_upper=np.array(upper),
            is_reference=np.array(refs),
            n=np.array(counts),
            n_kind="units",
            alpha=self.alpha,
            vcov=self._covariance(np.column_stack(contributions)),
            vcov_index=np.array(horizons),
            source="ChainedDiD",
            base_period="universal",
            anticipation=0,
            reference_event_times=tuple(sorted({references[g] - g for g in references})),
            estimand="ATT(g,t) after treatment; direct unrestricted differences before reference",
            post_periods=tuple(e for e in horizons if e >= 0),
        )

    def summary(self, alpha: Optional[float] = None) -> str:
        """Describe the fixed fit-time target and pointwise inference."""
        _require_fit_alpha(alpha, self.alpha, message=_SUMMARY_ALPHA_MESSAGE)
        return (
            f"Chained DiD (identity GMM, PT-post, never-treated controls)\n"
            f"ATT: {self.att:.6g}, SE: {self.se:.6g}, units: {self.n_units}\n"
            f"Pointwise normal intervals, alpha={self.alpha}; no attrition correction."
        )

    def print_summary(self) -> None:
        """Print the fit-time summary."""
        print(self.summary())

    def to_dataframe(self, level: str = "group_time") -> pd.DataFrame:
        """Return per-cell records or the canonical aggregate table."""
        if level in self._AGGREGATE_SUPPORTED:
            return self.aggregate(level).to_dataframe()
        if level != "group_time":
            raise ValueError("level must be 'group_time', 'simple' or 'event_study'")
        return pd.DataFrame(
            [
                {
                    "group": g,
                    "time": t,
                    "att": cell.effect,
                    "se": cell.se,
                    "t_stat": cell.t_stat,
                    "p_value": cell.p_value,
                    "conf_int_lower": cell.conf_int[0],
                    "conf_int_upper": cell.conf_int[1],
                    "n_treated": cell.n_treated,
                    "n_control": cell.n_control,
                    "available": cell.skip_reason is None,
                    "status": cell.skip_reason or "ok",
                }
                for (g, t), cell in self.group_time_effects.items()
            ]
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise canonical inference and explicit cell support."""
        return {
            "att": self.att,
            "se": self.se,
            "t_stat": self.t_stat,
            "p_value": self.p_value,
            "conf_int": self.conf_int,
            "alpha": self.alpha,
            "n_units": self.n_units,
            "group_time_effects": self.to_dataframe().to_dict(orient="records"),
            "vcov_index": self.vcov_index,
            "vcov": self.vcov.tolist() if self.vcov is not None else None,
            "diagnostics": self.diagnostics,
        }
