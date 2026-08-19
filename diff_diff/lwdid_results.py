"""Results class for the LWDiD (Lee & Wooldridge 2025, 2026) estimator."""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from diff_diff.aggregation import AggregationMixin, AggregationResult
from diff_diff.results_base import BaseResults, EventStudyResults


# How the overall staggered standard error was obtained. Cohort effects that
# share control units are correlated, so the basis is reported rather than
# left implicit.
def _as_float(value: Any) -> float:
    """Coerce an optional numeric cell entry to float, mapping None to NaN."""
    return np.nan if value is None else float(value)


_INFERENCE_BASIS_LABELS = {
    "composite_regression": "composite regression (LW 2026 eq. 7.18/7.19)",
    "joint_influence_function": "joint influence function across cohort-time cells",
    "unavailable_matching": "unavailable (matching has no influence function)",
    "unavailable_degenerate_cells": "unavailable (degenerate cohort-time cells)",
    "unit_bootstrap": "unit-resampling bootstrap (params/vcov remain analytical)",
    "cluster_bootstrap": "cluster-resampling bootstrap (params/vcov remain analytical)",
}


def _json_native_key(key: Any) -> Any:
    """Convert a numpy scalar or datetime-like dict key to its native equivalent."""
    if isinstance(key, np.bool_):
        return bool(key)
    if isinstance(key, np.integer):
        return int(key)
    if isinstance(key, np.floating):
        return float(key)
    # pd.NaT is datetime-like but has no meaningful isoformat; keep the
    # same convention as _to_json_native (NaT -> None) for consistency.
    if key is pd.NaT:
        return None
    if isinstance(key, (datetime.date, datetime.datetime)):
        # covers pd.Timestamp (subclass of datetime.datetime)
        return key.isoformat()
    if isinstance(key, np.datetime64):
        return pd.Timestamp(key).isoformat()
    if isinstance(key, pd.Period):
        return str(key)  # e.g. "2020Q1", preserves frequency semantics
    return key


def _to_json_native(obj: Any) -> Any:
    """Recursively convert numpy types to JSON-serializable Python natives.

    numpy scalars become int/float/bool, ndarrays become nested lists,
    and dict/list/tuple containers are converted element-wise (dict keys
    included). NaN/inf floats are kept as-is (float semantics preserved).
    Datetime-like values (datetime.date/datetime.datetime incl. pd.Timestamp,
    np.datetime64) become ISO-8601 strings; pd.Period becomes str (e.g.
    "2020Q1") to preserve frequency semantics; pd.NaT becomes None so the
    output is always json.dumps-able.
    """
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if obj is pd.NaT:
        return None
    if isinstance(obj, (datetime.date, datetime.datetime)):
        # covers pd.Timestamp (subclass of datetime.datetime)
        return obj.isoformat()
    if isinstance(obj, np.datetime64):
        if pd.isna(obj):
            return None
        return pd.Timestamp(obj).isoformat()
    if isinstance(obj, pd.Period):
        return str(obj)  # e.g. "2020Q1", preserves frequency semantics
    if isinstance(obj, np.ndarray):
        return [_to_json_native(v) for v in obj.tolist()]
    if isinstance(obj, dict):
        return {_json_native_key(k): _to_json_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_native(v) for v in obj]
    return obj


@dataclass
class LWDiDResults(BaseResults, AggregationMixin):
    """Results from LWDiD.fit().

    Follows the diff-diff standard results interface. Holds the headline ATT
    estimate and inference for the common-timing case, or per-cohort effects
    and an overall weighted ATT for the staggered case.

    Parameters
    ----------
    att : float
        Average treatment effect on the treated.
    se : float
        Standard error of the ATT estimate.
    t_stat : float
        t-statistic (att / se).
    p_value : float
        Two-sided p-value.
    conf_int : tuple of float
        (lower, upper) confidence interval at level ``1 - alpha``.
    n_obs : int
        Total observations used in estimation.
    n_treated : int
        Number of treated units.
    n_control : int
        Number of control units.
    rolling : str
        Transformation method used ('demean', 'detrend', 'demeanq', or 'detrendq').
    estimation_method : str
        Estimation method ('reg', 'ipw', 'dr', or 'psm').
    vcov_type : str
        Variance family ('classical', 'hc1', 'hc2', or 'hc3').
    alpha : float
        Significance level used for confidence intervals.
    df_inference : int or None
        Degrees of freedom used for t-distribution inference.
    cluster_name : str or None
        Name of the cluster variable, if clustered.
    n_clusters : int or None
        Number of clusters, if clustered.
    cohort_effects : dict or None
        Per-cohort ATT results for staggered designs.
    params : ndarray or None
        All coefficient estimates from the regression.
    bse : ndarray or None
        All standard errors from the regression.
    vcov : ndarray or None
        Variance-covariance matrix.
    """

    # ------------------------------------------------------------------ #
    # Core inference fields                                               #
    # ------------------------------------------------------------------ #
    att: float
    se: float
    t_stat: float
    p_value: float
    conf_int: Tuple[float, float]

    # ------------------------------------------------------------------ #
    # Sample information                                                  #
    # ------------------------------------------------------------------ #
    n_obs: int
    n_treated: int
    n_control: int

    # ------------------------------------------------------------------ #
    # Method metadata                                                     #
    # ------------------------------------------------------------------ #
    rolling: str
    estimation_method: str
    vcov_type: str
    alpha: float
    df_inference: Optional[int] = None
    cluster_name: Optional[str] = None
    n_clusters: Optional[int] = None

    # ------------------------------------------------------------------ #
    # Fit provenance (estimand/inference-affecting configuration - review #
    # finding: serialized results could not reconstruct what was fitted)  #
    # ------------------------------------------------------------------ #
    control_group: Optional[str] = None
    n_bootstrap: int = 0
    seed: Optional[int] = None
    #: Propensity-score trim bound used by the ipw/dr/psm paths (None for
    #: estimation_method='reg', where no propensity model is fitted).
    pscore_trim: Optional[float] = None
    #: PSM matching settings (None unless estimation_method='psm'):
    #: {'pscore_trim', 'n_neighbors', 'caliper', 'with_replacement'}
    psm_config: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------ #
    # Staggered-specific (optional)                                       #
    # ------------------------------------------------------------------ #
    cohort_effects: Optional[Dict[Any, Dict]] = field(default=None, repr=False)
    cohort_time_effects: Optional[Dict[Tuple[Any, Any], Dict]] = field(default=None, repr=False)
    inference_basis: Optional[str] = None
    #: Complete-case tau_omega composite point, exposed as a diagnostic when
    #: complete-case drops prevented it from being ``.att`` (None otherwise).
    att_tau_omega_complete_case: Optional[float] = None
    #: Treated / control units dropped by the tau_omega complete-case
    #: resolution (0 when the composite path did not run or dropped none).
    n_composite_treated_dropped: int = 0
    n_composite_controls_dropped: int = 0

    # ------------------------------------------------------------------ #
    # Event study (Appendix D) fields                                     #
    # ------------------------------------------------------------------ #
    event_study_effects: Optional[Dict[int, Dict]] = field(default=None, repr=False)
    event_study_vcov: Optional[np.ndarray] = field(default=None, repr=False)
    event_study_vcov_index: Optional[np.ndarray] = field(default=None, repr=False)
    event_study_df: Optional[Dict[int, float]] = field(default=None, repr=False)
    reference_periods: Tuple[int, ...] = field(default_factory=tuple, repr=False)
    cband_method: Optional[str] = field(default=None, repr=False)
    cband_crit_value: Optional[float] = field(default=None, repr=False)
    cband_n_bootstrap: Optional[int] = field(default=None, repr=False)

    # ------------------------------------------------------------------ #
    # Full regression output (optional)                                   #
    # ------------------------------------------------------------------ #
    params: Optional[np.ndarray] = field(default=None, repr=False)
    bse: Optional[np.ndarray] = field(default=None, repr=False)
    vcov: Optional[np.ndarray] = field(default=None, repr=False)

    # ------------------------------------------------------------------ #
    # Cached RI/WCB results (optional)                                    #
    # ------------------------------------------------------------------ #
    _ri_result: Optional[Any] = field(default=None, repr=False)
    _wcb_result: Optional[Any] = field(default=None, repr=False)

    # ------------------------------------------------------------------ #
    # Properties                                                          #
    # ------------------------------------------------------------------ #
    @property
    def pvalue(self) -> float:
        """Alias for p_value (diff-diff API convention)."""
        return self.p_value

    @property
    def ci(self) -> Tuple[float, float]:
        """Alias for conf_int (diff-diff API convention)."""
        return self.conf_int

    @property
    def is_staggered(self) -> bool:
        """Whether this result comes from a staggered adoption design."""
        return self.cohort_effects is not None

    #: ``simple`` reports the estimand ``fit()`` already computed; it never
    #: recombines cohort effects, which would silently swap the composite
    #: regression's joint inference for a cohort-independence assumption.
    _AGGREGATE_SUPPORTED = ("simple", "event_study", "group")
    #: balance_e is REJECTED (round-5 review: it was accepted but ignored,
    #: silently returning the unbalanced event-study surface): LWDiD stores
    #: no per-cohort estimation kit from which a balanced-cohort sample and
    #: its joint influence-function covariance could be recomputed post fit.
    _AGGREGATE_BALANCE_E_TYPES = ()

    def _aggregate_validate_weights(self, weights: Optional[str]) -> None:
        if weights is not None:
            raise ValueError(
                "LWDiDResults.aggregate() does not accept a weights selector "
                f"(got {weights!r}); LWDiD weights cohort-time cells by their "
                "treated mass, which is fixed by the estimator."
            )

    def _aggregate_compute(
        self,
        level: str,
        *,
        weights: Optional[str],
        balance_e: Optional[int],
    ) -> Any:
        if level == "group" and not self.is_staggered:
            raise ValueError(
                "aggregate('group') is only available for staggered fits; a "
                "common-timing design has a single treatment cohort, so "
                "there is no group dimension to aggregate over."
            )

        if level == "simple":
            ci = self.conf_int
            return AggregationResult(
                level="simple",
                label=np.array(["overall"], dtype=object),
                target=np.array(["att"], dtype=object),
                att=np.array([self.att], dtype=float),
                se=np.array([self.se], dtype=float),
                t_stat=np.array([self.t_stat], dtype=float),
                p_value=np.array([self.p_value], dtype=float),
                conf_int_lower=np.array([ci[0]], dtype=float),
                conf_int_upper=np.array([ci[1]], dtype=float),
                n=np.array([float(self.n_treated)], dtype=float),
                df=np.array(
                    [np.nan if self.df_inference is None else float(self.df_inference)],
                    dtype=float,
                ),
                alpha=self.alpha,
                n_kind="units",
                weight=np.array([1.0], dtype=float),
                estimator="LWDiD",
            )

        if level == "group":
            cohorts = list(self.cohort_effects or {})
            effects = [self.cohort_effects[g] for g in cohorts]  # type: ignore[index]

            def _column(key: str, default: float = np.nan) -> np.ndarray:
                return np.array([_as_float(e.get(key, default)) for e in effects], dtype=float)

            bounds = [e.get("conf_int", (np.nan, np.nan)) for e in effects]
            return AggregationResult(
                level="group",
                label=np.array(cohorts, dtype=object),
                target=np.array(["att"] * len(cohorts), dtype=object),
                att=_column("att"),
                se=_column("se"),
                t_stat=_column("t_stat"),
                p_value=_column("p_value"),
                conf_int_lower=np.array([_as_float(b[0]) for b in bounds], dtype=float),
                conf_int_upper=np.array([_as_float(b[1]) for b in bounds], dtype=float),
                n=_column("n_treated"),
                df=_column("df"),
                alpha=self.alpha,
                n_kind="units",
                weight=_column("weight"),
                estimator="LWDiD",
            )

        if level == "event_study":
            es_effects = self.event_study_effects or {}
            reference_periods = set(self.reference_periods or ())
            labels = sorted(set(es_effects) | reference_periods)
            rows = [es_effects.get(label, {}) for label in labels]
            is_reference = np.array([label in reference_periods for label in labels], dtype=bool)
            att = np.array(
                [
                    row.get("effect", 0.0 if reference else np.nan)
                    for row, reference in zip(rows, is_reference)
                ],
                dtype=float,
            )
            se = np.array([row.get("se", np.nan) for row in rows], dtype=float)
            t_stat = np.array([row.get("t_stat", np.nan) for row in rows], dtype=float)
            p_value = np.array([row.get("p_value", np.nan) for row in rows], dtype=float)
            ci_lower = np.array(
                [row.get("conf_int", (np.nan, np.nan))[0] for row in rows], dtype=float
            )
            ci_upper = np.array(
                [row.get("conf_int", (np.nan, np.nan))[1] for row in rows], dtype=float
            )
            n = np.array([row.get("n_treated", np.nan) for row in rows], dtype=float)
            cband_lower = np.array(
                [row.get("cband_conf_int", (np.nan, np.nan))[0] for row in rows], dtype=float
            )
            cband_upper = np.array(
                [row.get("cband_conf_int", (np.nan, np.nan))[1] for row in rows], dtype=float
            )
            has_band = any(np.isfinite(cband_lower) & np.isfinite(cband_upper))
            vcov = self.event_study_vcov if self.event_study_vcov is not None else None
            vcov_index = (
                self.event_study_vcov_index if self.event_study_vcov_index is not None else None
            )
            has_vcov = vcov is not None and vcov_index is not None and len(vcov_index) > 0
            df = None
            if self.event_study_df is not None:
                df = np.array([self.event_study_df.get(label, np.nan) for label in labels])
            return EventStudyResults(
                event_time=np.array(labels),
                att=att,
                se=se,
                t_stat=t_stat,
                p_value=p_value,
                conf_int_lower=ci_lower,
                conf_int_upper=ci_upper,
                is_reference=is_reference,
                n=n,
                n_kind="units",
                time_scale="relative",
                event_time_convention="e0_first_treated",
                vcov=vcov if has_vcov else None,
                vcov_index=vcov_index if has_vcov else None,
                cband_lower=cband_lower if has_band else None,
                cband_upper=cband_upper if has_band else None,
                cband_crit_value=self.cband_crit_value,
                alpha=self.alpha,
                source="LWDiDResults",
                df=df,
                # Scalar-df provenance, mirroring results_base's resolution
                # rule: no survey notion, so the bare df_inference carrier.
                df_survey=None if self.df_inference is None else float(self.df_inference),
            )

        raise ValueError(f"Unsupported aggregation method: {level!r}")

    # ------------------------------------------------------------------ #
    # Serialization                                                       #
    # ------------------------------------------------------------------ #
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            For common timing: a single-row DataFrame.
            For staggered: one row per cohort plus an "Overall" row.
        """
        if not self.is_staggered:
            rows: List[Dict[str, Any]] = [
                {
                    "term": "ATT",
                    "att": self.att,
                    "se": self.se,
                    "t_stat": self.t_stat,
                    "p_value": self.p_value,
                    "ci_lower": self.conf_int[0],
                    "ci_upper": self.conf_int[1],
                    "n_obs": self.n_obs,
                    "n_treated": self.n_treated,
                    "n_control": self.n_control,
                    "rolling": self.rolling,
                    "estimation_method": self.estimation_method,
                    "vcov_type": self.vcov_type,
                }
            ]
            return pd.DataFrame(rows)

        rows_stag: List[Dict[str, Any]] = []
        for cohort, eff in self.cohort_effects.items():  # type: ignore[union-attr]
            ci = eff.get("conf_int", (np.nan, np.nan))
            n_t = eff.get("n_treated", 0)
            n_c = eff.get("n_control", 0)
            rows_stag.append(
                {
                    "cohort": cohort,
                    "att": eff.get("att", np.nan),
                    "se": eff.get("se", np.nan),
                    "t_stat": eff.get("t_stat", np.nan),
                    "p_value": eff.get("p_value", np.nan),
                    "ci_lower": ci[0] if ci else np.nan,
                    "ci_upper": ci[1] if ci else np.nan,
                    "n_treated": n_t,
                    "n_control": n_c,
                    "rolling": self.rolling,
                    "estimation_method": self.estimation_method,
                    "vcov_type": self.vcov_type,
                }
            )
        # Append overall row
        rows_stag.append(
            {
                "cohort": "Overall",
                "att": self.att,
                "se": self.se,
                "t_stat": self.t_stat,
                "p_value": self.p_value,
                "ci_lower": self.conf_int[0],
                "ci_upper": self.conf_int[1],
                "n_treated": self.n_treated,
                "n_control": self.n_control,
                "rolling": self.rolling,
                "estimation_method": self.estimation_method,
                "vcov_type": self.vcov_type,
            }
        )
        return pd.DataFrame(rows_stag)

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to a JSON-serializable dictionary.

        Returns
        -------
        dict
            All scalar results and metadata. Arrays are converted to lists
            and numpy scalars (including nested dict values and keys) to
            native Python types, so ``json.dumps(result.to_dict())`` works
            directly.
        """
        result: Dict[str, Any] = {
            "att": self.att,
            "se": self.se,
            "t_stat": self.t_stat,
            "p_value": self.p_value,
            "conf_int_lower": self.conf_int[0],
            "conf_int_upper": self.conf_int[1],
            "n_obs": self.n_obs,
            "n_treated": self.n_treated,
            "n_control": self.n_control,
            "rolling": self.rolling,
            "estimation_method": self.estimation_method,
            "vcov_type": self.vcov_type,
            "alpha": self.alpha,
        }
        if self.cluster_name is not None:
            result["cluster_name"] = self.cluster_name
        if self.n_clusters is not None:
            result["n_clusters"] = self.n_clusters
        if self.cohort_effects is not None:
            result["cohort_effects"] = {str(k): v for k, v in self.cohort_effects.items()}
        if self.cohort_time_effects is not None:
            result["cohort_time_effects"] = {
                f"{g},{t}": value for (g, t), value in self.cohort_time_effects.items()
            }
        if self.inference_basis is not None:
            result["inference_basis"] = self.inference_basis
        if self.df_inference is not None:
            result["df_inference"] = self.df_inference
        if self.control_group is not None:
            result["control_group"] = self.control_group
        result["n_bootstrap"] = self.n_bootstrap
        if self.seed is not None:
            result["seed"] = self.seed
        if self.pscore_trim is not None:
            result["pscore_trim"] = self.pscore_trim
        if self.psm_config is not None:
            result["psm_config"] = dict(self.psm_config)
        if self.att_tau_omega_complete_case is not None:
            result["att_tau_omega_complete_case"] = self.att_tau_omega_complete_case
        if self.n_composite_treated_dropped or self.n_composite_controls_dropped:
            result["n_composite_treated_dropped"] = self.n_composite_treated_dropped
            result["n_composite_controls_dropped"] = self.n_composite_controls_dropped
        if self.params is not None:
            result["params"] = self.params.tolist()
        if self.bse is not None:
            result["bse"] = self.bse.tolist()
        if self.event_study_effects is not None:
            result["event_study_effects"] = {str(k): v for k, v in self.event_study_effects.items()}
            result["reference_periods"] = list(self.reference_periods)
            result["cband_method"] = self.cband_method
            result["cband_crit_value"] = self.cband_crit_value
            result["cband_n_bootstrap"] = self.cband_n_bootstrap
        return _to_json_native(result)

    # ------------------------------------------------------------------ #
    # Aggregation                                                         #
    # ------------------------------------------------------------------ #
    def to_csv(self, path: str) -> None:
        """Export results to CSV file.

        Parameters
        ----------
        path : str
            File path for the CSV output.
        """
        self.to_dataframe().to_csv(path, index=False)

    # ------------------------------------------------------------------ #
    # Text summary                                                        #
    # ------------------------------------------------------------------ #
    def summary(self) -> str:
        """Formatted text summary of results.

        Returns
        -------
        str
            Human-readable summary table.
        """
        from diff_diff.results import _format_vcov_label, _get_significance_stars

        ci_pct = int(round((1 - self.alpha) * 100))
        width = 88
        bar = "=" * width
        dash = "-" * width

        def _fmt(x: Any, nd: int = 4) -> str:
            try:
                xf = float(x)
            except (TypeError, ValueError):
                return ""
            return "" if np.isnan(xf) else f"{xf:.{nd}f}"

        lines: List[str] = [
            bar,
            "Lee & Wooldridge DiD (LWDiD) Results".center(width),
            bar,
            f"Observations: {self.n_obs}    "
            f"Treated units: {self.n_treated}    "
            f"Control units: {self.n_control}",
            f"Rolling: {self.rolling}    "
            f"Method: {self.estimation_method}    "
            f"Alpha: {self.alpha}",
        ]

        # Variance label
        vcov_label = _format_vcov_label(
            self.vcov_type,
            cluster_name=self.cluster_name,
            n_clusters=self.n_clusters,
            n_obs=self.n_obs,
        )
        if vcov_label:
            lines.append(f"Std. errors: {vcov_label}")

        # Header for results table
        header = (
            f"{'':>12}  {'Estimate':>10}  {'Std.Err':>10}  {'t':>8}  "
            f"{'P>|t|':>8}  [{ci_pct}% Conf. Int.]"
        )

        # Main ATT row
        lines.append("")
        if self.is_staggered:
            lines.append("Cohort-level effects:")
            lines.append(dash)
            lines.append(header)
            lines.append(dash)
            for cohort, eff in self.cohort_effects.items():  # type: ignore[union-attr]
                ci = eff.get("conf_int", (np.nan, np.nan))
                p = eff.get("p_value", np.nan)
                stars = "" if np.isnan(p) else _get_significance_stars(float(p))
                label = f"G={cohort}"
                lines.append(
                    f"{label:>12}  {_fmt(eff.get('att')):>10}  "
                    f"{_fmt(eff.get('se')):>10}  "
                    f"{_fmt(eff.get('t_stat'), 2):>8}  "
                    f"{_fmt(p, 3):>8}  "
                    f"[{_fmt(ci[0]):>9}, {_fmt(ci[1]):>9}] {stars}"
                )
            lines.append(dash)
            # Overall ATT
            stars = _get_significance_stars(self.p_value) if not np.isnan(self.p_value) else ""
            lines.append(
                f"{'Overall ATT':>12}  {_fmt(self.att):>10}  "
                f"{_fmt(self.se):>10}  "
                f"{_fmt(self.t_stat, 2):>8}  "
                f"{_fmt(self.p_value, 3):>8}  "
                f"[{_fmt(self.conf_int[0]):>9}, {_fmt(self.conf_int[1]):>9}] {stars}"
            )
        else:
            lines.append("ATT estimate:")
            lines.append(dash)
            lines.append(header)
            lines.append(dash)
            stars = _get_significance_stars(self.p_value) if not np.isnan(self.p_value) else ""
            lines.append(
                f"{'ATT':>12}  {_fmt(self.att):>10}  "
                f"{_fmt(self.se):>10}  "
                f"{_fmt(self.t_stat, 2):>8}  "
                f"{_fmt(self.p_value, 3):>8}  "
                f"[{_fmt(self.conf_int[0]):>9}, {_fmt(self.conf_int[1]):>9}] {stars}"
            )

        lines.append(bar)
        if self.inference_basis is not None:
            label = _INFERENCE_BASIS_LABELS.get(self.inference_basis, self.inference_basis)
            lines.append(f"Overall inference: {label}")
        lines.append("Signif. codes: *** p<0.001, ** p<0.01, * p<0.05")
        return "\n".join(lines)

    def print_summary(self) -> None:
        """Print the formatted summary to stdout."""
        print(self.summary())

    # ================================================================
    # Advanced inference and diagnostics (delegate to standalone modules)
    # ================================================================

    @property
    def ri_pvalue(self):
        """Randomization inference p-value (None if not computed)."""
        if self._ri_result is not None:
            return self._ri_result.pvalue
        return None

    @property
    def bootstrap_pvalue(self):
        """Wild cluster bootstrap p-value (None if not computed)."""
        if self._wcb_result is not None:
            return self._wcb_result.p_value
        return None

    def _replay_arrays(self, method_name):
        """Fitted-sample arrays for the post-fit advanced-inference methods.

        Round-5 review: these methods previously accepted arbitrary caller
        arrays and fit a non-interacted design, so the cached p-values
        could describe a DIFFERENT estimand than ``.att`` (measured on a
        covariate-unbalanced RA fit: fitted 3.98 vs tested 3.26). They now
        REPLAY the fit-time collapsed cross-section and the exact RA
        design; no data arguments are accepted.
        """
        spec = getattr(self, "_replay_spec", None)
        if spec is None:
            raise ValueError(
                f"{method_name} replays the fitted common-timing estimation "
                "sample, which this results object does not carry (staggered "
                "and degenerate fits are not supported). Use the standalone "
                "module function with explicit arrays instead."
            )
        if self.estimation_method != "reg":
            raise ValueError(
                f"{method_name} replays the fitted RA regression and is only "
                f"defined for estimation_method='reg' (got "
                f"'{self.estimation_method}'): re-estimating the "
                f"{self.estimation_method} estimator per draw is not "
                "implemented. Use the standalone module function on arrays "
                "of your choosing (a generic [1, D, X] contrast, NOT the "
                "fitted estimand)."
            )
        return spec

    @staticmethod
    def _fit_used_interactions(treatment, controls):
        """Mirror _estimate_reg's LW eq. 3.3 gate: interactions require
        N_1 > K+1 and N_0 > K+1; otherwise the fit used plain (1, D, X)
        (round-7 review: the replay always interacted, so small-arm fits'
        replayed statistic mismatched .att and the coherence assert made
        their post-fit inference unusable)."""
        if controls is None:
            return False
        n_treated = int((treatment == 1).sum())
        n_control = len(treatment) - n_treated
        # IDENTIFIED control dimension, mirroring _estimate_reg exactly
        # (round-11 review: the nominal column count diverged from the
        # fit's gate under collinear controls).
        from diff_diff.linalg import _detect_rank_deficiency

        k = int(
            _detect_rank_deficiency(np.column_stack([np.ones(len(treatment)), controls]))[0] - 1
        )
        return n_treated > k + 1 and n_control > k + 1

    @classmethod
    def _replay_controls(cls, treatment, controls):
        """The exact auxiliary columns the fitted RA regression used:
        [X, D*(X - Xbar_1)] when the interaction gate held, plain X
        otherwise (LW eq. E.1 / eq. 3.3)."""
        if controls is None:
            return None
        if not cls._fit_used_interactions(treatment, controls):
            return controls
        xbar1 = controls[treatment == 1].mean(axis=0)
        return np.column_stack([controls, treatment[:, None] * (controls - xbar1)])

    def _assert_replay_coherent(self, observed, method_name):
        if not np.isclose(observed, self.att, rtol=1e-8, atol=1e-10):
            raise RuntimeError(
                f"{method_name}: the replayed observed ATT ({observed!r}) "
                f"does not match the fitted .att ({self.att!r}); refusing to "
                "cache inference for a different estimand (fail closed)."
            )

    def wild_cluster_bootstrap(
        self,
        *,
        n_bootstrap=999,
        weight_type="rademacher",
        alpha=None,
        seed=None,
    ):
        """Run wild cluster bootstrap inference on the fitted estimation sample.

        Replays the fit-time collapsed cross-section and the exact fitted
        RA design (intercept, treatment, covariates, and the treatment-
        centered interactions) through the house WCR engine
        (test-inversion CI, CR1 se, strict-exceedance p-value); the
        observed coefficient is asserted equal to ``.att`` before caching.
        Requires a clustered (``cluster=``), common-timing,
        ``estimation_method='reg'`` fit. ``alpha=None`` inherits the
        fitted confidence level. Result is cached and accessible via the
        ``bootstrap_pvalue`` property.
        """
        from diff_diff.lwdid_wild_bootstrap import wild_cluster_bootstrap as _wcb

        spec = self._replay_arrays("wild_cluster_bootstrap")
        if spec["cluster_ids"] is None:
            raise ValueError(
                "wild_cluster_bootstrap requires a clustered fit: construct "
                "the estimator with cluster= and refit."
            )
        result = _wcb(
            spec["y"],
            spec["treatment"],
            spec["cluster_ids"],
            self._replay_controls(spec["treatment"], spec["controls"]),
            n_bootstrap=n_bootstrap,
            weight_type=weight_type,
            alpha=self.alpha if alpha is None else alpha,
            seed=seed,
        )
        self._assert_replay_coherent(result.att, "wild_cluster_bootstrap")
        object.__setattr__(self, "_wcb_result", result)
        return result

    def randomization_test(self, *, n_reps=1000, method="permutation", seed=None):
        """Run Fisher randomization inference on the fitted estimation sample.

        Replays the fit-time collapsed cross-section; with covariates the
        RA design's treated covariate mean and interaction columns are
        RECOMPUTED for every permuted assignment (``design='ra_interacted'``),
        so each permutation tests the same estimator the fit reported. The
        observed statistic is asserted equal to ``.att`` before caching.
        Requires a common-timing ``estimation_method='reg'`` fit. Result is
        cached and accessible via the ``ri_pvalue`` property.
        """
        from diff_diff.lwdid_randomization import randomization_inference as _ri

        spec = self._replay_arrays("randomization_test")
        # Match the fitted design exactly: 'ra_interacted' only when the
        # fit's interaction gate held (round-7 review). NOTE the permuted
        # draws under the plain design keep plain (1, D, X) too - each
        # permutation tests the same estimator the fit reported.
        design = (
            "ra_interacted"
            if self._fit_used_interactions(spec["treatment"], spec["controls"])
            else "linear"
        )
        result = _ri(
            spec["y"],
            spec["treatment"],
            spec["controls"],
            n_reps=n_reps,
            method=method,
            seed=seed,
            design=design,
        )
        self._assert_replay_coherent(result.att_observed, "randomization_test")
        object.__setattr__(self, "_ri_result", result)
        return result

    # ------------------------------------------------------------------ #
    # Repr                                                                #
    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:
        cluster = f", cluster={self.cluster_name}, G={self.n_clusters}" if self.cluster_name else ""
        att_s = "nan" if np.isnan(self.att) else f"{self.att:.4f}"
        se_s = "nan" if np.isnan(self.se) else f"{self.se:.4f}"
        stag = ", staggered=True" if self.is_staggered else ""
        return (
            f"LWDiDResults("
            f"ATT={att_s}, SE={se_s}, "
            f"rolling={self.rolling!r}, estimation_method={self.estimation_method!r}, "
            f"vcov_type={self.vcov_type!r}{cluster}{stag})"
        )
