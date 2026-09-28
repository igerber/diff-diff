"""
Result container classes for Continuous Difference-in-Differences estimator.

Provides dataclass containers for dose-response curves, group-time effects,
and aggregated estimation results.
"""

import dataclasses
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from diff_diff.aggregation import AggregationMixin, AggregationResult
from diff_diff.continuous_did_aggregation import _ContinuousDiDAggregationMixin
from diff_diff.results import _format_survey_block, _get_significance_stars
from diff_diff.results_base import (
    _SUMMARY_ALPHA_MESSAGE,
    BaseResults,
    _coverage_pct,
    _require_fit_alpha,
    build_event_study_surface,
)
from diff_diff.utils import safe_inference

__all__ = ["ContinuousDiDResults", "DoseResponseCurve"]


class _ContinuousKitAggregator(_ContinuousDiDAggregationMixin):
    """Throwaway host for the post-fit event-study recompute (row M-025).

    A fresh instance runs each ``aggregate('event_study')`` call so the
    recompute can never read or write estimator/results state. Sets
    exactly the mixin's host-attribute contract: ``alpha``.
    """

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha


@dataclass
class DoseResponseCurve:
    """
    Dose-response curve from continuous DiD estimation.

    Attributes
    ----------
    dose_grid : np.ndarray
        Evaluation points, shape ``(n_grid,)``.
    effects : np.ndarray
        ATT(d) or ACRT(d) values, shape ``(n_grid,)``.
    se : np.ndarray
        Standard errors, shape ``(n_grid,)``.
    conf_int_lower : np.ndarray
        Lower CI bounds, shape ``(n_grid,)``.
    conf_int_upper : np.ndarray
        Upper CI bounds, shape ``(n_grid,)``.
    target : str
        ``"att"`` or ``"acrt"``.
    """

    dose_grid: np.ndarray
    effects: np.ndarray
    se: np.ndarray
    conf_int_lower: np.ndarray
    conf_int_upper: np.ndarray
    target: str
    p_value: Optional[np.ndarray] = None
    n_bootstrap: int = 0
    df_survey: Optional[int] = None

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame with dose, effect, se, CI, t_stat, p_value."""
        n = len(self.effects)
        if self.n_bootstrap > 0 and self.p_value is not None:
            # Bootstrap inference: use stored p-values, t-stat is undefined
            t_stat = np.full(n, np.nan)
            p_value = self.p_value
        else:
            # Analytic inference: compute t-stat and p-value from normal approx
            from diff_diff.utils import safe_inference

            t_stat = np.full(n, np.nan)
            p_value = np.full(n, np.nan)
            for i in range(n):
                t_i, p_i, _ = safe_inference(self.effects[i], self.se[i], df=self.df_survey)
                t_stat[i] = t_i
                p_value[i] = p_i
        return pd.DataFrame(
            {
                "dose": self.dose_grid,
                "effect": self.effects,
                "se": self.se,
                "conf_int_lower": self.conf_int_lower,
                "conf_int_upper": self.conf_int_upper,
                "t_stat": t_stat,
                "p_value": p_value,
            }
        )


@dataclass
class ContinuousDiDResults(BaseResults, AggregationMixin):
    """
    Results from Continuous Difference-in-Differences estimation.

    Implements Callaway, Goodman-Bacon & Sant'Anna (2024).

    Attributes
    ----------
    dose_response_att : DoseResponseCurve
        ATT(d) dose-response curve.
    dose_response_acrt : DoseResponseCurve
        ACRT(d) dose-response curve.
    overall_att : float
        Binarized overall ATT (ATT^{loc} under PT, equals ATT^{glob} under SPT).
    overall_acrt : float
        Plug-in overall ACRT^{glob}.
    group_time_effects : dict
        Per (g,t) cell results.
    base_period : str
        Base period strategy (``"varying"`` or ``"universal"``).
    anticipation : int
        Number of anticipation periods.
    n_bootstrap : int
        Number of bootstrap iterations used.
    bootstrap_weights : str
        Bootstrap weight type (``"rademacher"``, ``"mammen"``, or ``"webb"``).
    seed : int or None
        Random seed used for bootstrap.
    rank_deficient_action : str
        How rank deficiency is handled (``"warn"``, ``"error"``, ``"silent"``).
    event_study_df : float or None
        Scalar survey df governing the event-study rows' t-inference.
        ``None`` on non-survey fits, on bootstrapped fits, when no
        fit-time event-study surface was built, and for the
        replicate-undefined ``0`` sentinel.
    """

    dose_response_att: DoseResponseCurve
    dose_response_acrt: DoseResponseCurve
    overall_att: float
    overall_att_se: float
    overall_att_t_stat: float
    overall_att_p_value: float
    overall_att_conf_int: Tuple[float, float]
    overall_acrt: float
    overall_acrt_se: float
    overall_acrt_t_stat: float
    overall_acrt_p_value: float
    overall_acrt_conf_int: Tuple[float, float]
    group_time_effects: Dict[Tuple[Any, Any], Dict[str, Any]]
    dose_grid: np.ndarray
    groups: List[Any]
    time_periods: List[Any]
    n_obs: int
    n_treated_units: int
    n_control_units: int
    alpha: float = 0.05
    control_group: str = "never_treated"
    degree: int = 3
    num_knots: int = 0
    base_period: str = "varying"
    anticipation: int = 0
    n_bootstrap: int = 0
    bootstrap_weights: str = "rademacher"
    seed: Optional[int] = None
    rank_deficient_action: str = "warn"
    # Covariate adjustment (conditional parallel trends). ``covariates`` is None
    # for the unconditional path; ``estimation_method`` is only meaningful when
    # covariates are used (``"reg"`` or ``"dr"``).
    covariates: Optional[List[str]] = field(default=None)
    estimation_method: str = "dr"
    pscore_trim: float = 0.01
    epv_threshold: float = 10.0
    pscore_fallback: str = "error"
    # "continuous" (B-spline sieve dose-response) or "discrete" (saturated
    # per-dose-level regression); the ``dose_grid`` holds the distinct dose
    # levels when discrete.
    treatment_type: str = "continuous"
    # Lowest-dose reference d_L for ``control_group="lowest_dose"`` (Remark 3.1);
    # the estimand is ``ATT(d) - ATT(d_L)`` and ``ATT(d_L) = 0`` by construction.
    # ``None`` for the never/not-yet-treated (D=0 control) paths.
    reference_dose: Optional[float] = None
    event_study_effects: Optional[Dict[int, Dict[str, Any]]] = field(default=None)
    # Survey design metadata (SurveyMetadata instance from diff_diff.survey)
    survey_metadata: Optional[Any] = field(default=None)
    # Post-fit aggregation kit (row M-025), attached by ContinuousDiD.fit().
    # New fields are appended AFTER this one (positional-__init__
    # compatibility). Only the 'event_study' recompute reads it;
    # 'simple'/'dose' are views.
    _aggregation_kit: Optional[Any] = field(default=None, repr=False, compare=False)
    # Scalar survey df governing the event-study rows' t-inference. None on
    # non-survey fits, on bootstrapped fits (percentile inference; the ES
    # recompute also fails closed there), when no event-study surface was
    # built, and for the replicate-undefined 0 sentinel. Appended last per
    # the positional-__init__ convention above.
    event_study_df: Optional[float] = None

    # Post-fit aggregation routing (M-122 contract). ContinuousDiD's extra
    # 'dose' level is documented in the ledger row and v4-design section 6;
    # no level takes balance_e (the estimator has no balance_e machinery).
    _AGGREGATE_SUPPORTED: ClassVar[Tuple[str, ...]] = ("simple", "event_study", "dose")
    _AGGREGATE_BALANCE_E_TYPES: ClassVar[Tuple[str, ...]] = ()

    # --- Inference-field aliases (balance/external-adapter compatibility) ---
    # ATT-side is the headline contract; ACRT remains accessible via overall_acrt_*.
    @property
    def att(self) -> float:
        return self.overall_att

    @property
    def se(self) -> float:
        return self.overall_att_se

    @property
    def conf_int(self) -> Tuple[float, float]:
        return self.overall_att_conf_int

    @property
    def p_value(self) -> float:
        return self.overall_att_p_value

    @property
    def t_stat(self) -> float:
        return self.overall_att_t_stat

    # `overall_*` aliases for naming consistency with the rest of the staggered family.
    @property
    def overall_se(self) -> float:
        return self.overall_att_se

    @property
    def overall_conf_int(self) -> Tuple[float, float]:
        return self.overall_att_conf_int

    @property
    def overall_p_value(self) -> float:
        return self.overall_att_p_value

    @property
    def overall_t_stat(self) -> float:
        return self.overall_att_t_stat

    def __repr__(self) -> str:
        sig_att = _get_significance_stars(self.overall_att_p_value)
        sig_acrt = _get_significance_stars(self.overall_acrt_p_value)
        return (
            f"ContinuousDiDResults("
            f"ATT_glob={self.overall_att:.4f}{sig_att}, "
            f"ACRT_glob={self.overall_acrt:.4f}{sig_acrt}, "
            f"n_groups={len(self.groups)}, "
            f"n_periods={len(self.time_periods)})"
        )

    @property
    def coef_var(self) -> float:
        """Coefficient of variation: SE / abs(overall ATT). NaN when ATT is 0 or SE non-finite."""
        if not (np.isfinite(self.overall_att_se) and self.overall_att_se >= 0):
            return np.nan
        if not np.isfinite(self.overall_att) or self.overall_att == 0:
            return np.nan
        return self.overall_att_se / abs(self.overall_att)

    def summary(self, alpha: Optional[float] = None) -> str:
        """Generate formatted summary.

        Parameters
        ----------
        alpha : float, optional
            Accepted for signature uniformity. The stored intervals were
            computed at fit time; a value different from the stored
            ``alpha`` raises ValueError rather than silently recomputing
            or relabeling. Re-fit at the desired alpha instead.
        """
        alpha = _require_fit_alpha(alpha, self.alpha, message=_SUMMARY_ALPHA_MESSAGE)
        conf_level = _coverage_pct(alpha)
        w = 85

        lines = [
            "=" * w,
            "Continuous Difference-in-Differences Results".center(w),
            "(Callaway, Goodman-Bacon & Sant'Anna 2024)".center(w),
            "=" * w,
            "",
            f"{'Total observations:':<30} {self.n_obs:>10}",
            f"{'Treated units:':<30} {self.n_treated_units:>10}",
            f"{'Control units:':<30} {self.n_control_units:>10}",
            f"{'Treatment cohorts:':<30} {len(self.groups):>10}",
            f"{'Time periods:':<30} {len(self.time_periods):>10}",
            f"{'Control group:':<30} {self.control_group:>10}",
            f"{'Treatment type:':<30} {self.treatment_type:>10}",
        ]
        # Lowest-dose reference (Remark 3.1): show d_L when it is the control.
        if self.reference_dose is not None:
            lines.append(f"{'Reference dose (d_L):':<30} {self.reference_dose:>10.4g}")
        # Basis metadata: B-spline degree/knots (continuous) or the number of
        # saturated dose levels (discrete).
        if self.treatment_type == "discrete":
            lines.append(f"{'Dose levels:':<30} {len(self.dose_grid):>10}")
        else:
            lines.append(f"{'B-spline degree:':<30} {self.degree:>10}")
            lines.append(f"{'Interior knots:':<30} {self.num_knots:>10}")
        lines.append(f"{'Base period:':<30} {self.base_period:>10}")
        lines.append(f"{'Anticipation:':<30} {self.anticipation:>10}")
        if self.covariates:
            lines.append(f"{'Covariates:':<30} {', '.join(self.covariates):>10}")
            lines.append(f"{'Estimation method:':<30} {self.estimation_method:>10}")
        lines.append("")

        # Add survey design info
        if self.survey_metadata is not None:
            sm = self.survey_metadata
            lines.extend(_format_survey_block(sm, w))

        # Overall summary parameters
        lines.extend(
            [
                "-" * w,
                "Overall Summary Parameters".center(w),
                "-" * w,
                f"{'Parameter':<15} {'Estimate':>12} {'Std. Err.':>12} "
                f"{'t-stat':>10} {'P>|t|':>10} {'Sig.':>6}",
                "-" * w,
            ]
        )
        for label, est, se, t, p in [
            (
                "ATT_glob",
                self.overall_att,
                self.overall_att_se,
                self.overall_att_t_stat,
                self.overall_att_p_value,
            ),
            (
                "ACRT_glob",
                self.overall_acrt,
                self.overall_acrt_se,
                self.overall_acrt_t_stat,
                self.overall_acrt_p_value,
            ),
        ]:
            t_str = f"{t:>10.3f}" if np.isfinite(t) else f"{'NaN':>10}"
            p_str = f"{p:>10.4f}" if np.isfinite(p) else f"{'NaN':>10}"
            sig = _get_significance_stars(p)
            lines.append(f"{label:<15} {est:>12.4f} {se:>12.4f} {t_str} {p_str} {sig:>6}")
        lines.extend(
            [
                "-" * w,
                "",
                f"{conf_level}% CI for ATT_glob: "
                f"[{self.overall_att_conf_int[0]:.4f}, {self.overall_att_conf_int[1]:.4f}]",
                f"{conf_level}% CI for ACRT_glob: "
                f"[{self.overall_acrt_conf_int[0]:.4f}, {self.overall_acrt_conf_int[1]:.4f}]",
            ]
        )

        cv = self.coef_var
        if np.isfinite(cv):
            lines.append(f"{'CV (SE/abs(ATT)):':<25} {cv:>10.4f}")

        lines.append("")

        # Dose-response curve summary (first/mid/last points)
        if len(self.dose_grid) > 0:
            lines.extend(
                [
                    "-" * w,
                    "Dose-Response Curve (selected points)".center(w),
                    "-" * w,
                    f"{'Dose':>10} {'ATT(d)':>12} {'SE':>10} " f"{'ACRT(d)':>12} {'SE':>10}",
                    "-" * w,
                ]
            )
            n_grid = len(self.dose_grid)
            indices = sorted(set([0, n_grid // 4, n_grid // 2, 3 * n_grid // 4, n_grid - 1]))
            for idx in indices:
                if idx < n_grid:
                    lines.append(
                        f"{self.dose_grid[idx]:>10.3f} "
                        f"{self.dose_response_att.effects[idx]:>12.4f} "
                        f"{self.dose_response_att.se[idx]:>10.4f} "
                        f"{self.dose_response_acrt.effects[idx]:>12.4f} "
                        f"{self.dose_response_acrt.se[idx]:>10.4f}"
                    )
            lines.extend(["-" * w, ""])

        # Event study effects if available
        if self.event_study_effects:
            lines.extend(
                [
                    "-" * w,
                    "Event Study (Dynamic) Effects (Binarized ATT)".center(w),
                    "-" * w,
                    f"{'Rel. Period':<15} {'Estimate':>12} {'Std. Err.':>12} "
                    f"{'t-stat':>10} {'P>|t|':>10} {'Sig.':>6}",
                    "-" * w,
                ]
            )
            for rel_t in sorted(self.event_study_effects.keys()):
                eff = self.event_study_effects[rel_t]
                sig = _get_significance_stars(eff["p_value"])
                t_str = f"{eff['t_stat']:>10.3f}" if np.isfinite(eff["t_stat"]) else f"{'NaN':>10}"
                p_str = (
                    f"{eff['p_value']:>10.4f}" if np.isfinite(eff["p_value"]) else f"{'NaN':>10}"
                )
                lines.append(
                    f"{rel_t:<15} {eff['effect']:>12.4f} {eff['se']:>12.4f} "
                    f"{t_str} {p_str} {sig:>6}"
                )
            lines.extend(["-" * w, ""])

        lines.extend(
            [
                "Signif. codes: '***' 0.001, '**' 0.01, '*' 0.05, '.' 0.1",
                "=" * w,
            ]
        )
        return "\n".join(lines)

    def print_summary(self, alpha: Optional[float] = None) -> None:
        """Print summary to stdout."""
        print(self.summary(alpha))

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert headline results to a dictionary.

        Returns
        -------
        Dict[str, Any]
            Canonical ATT inference row, the ACRT companion estimand, and
            scalar metadata. Detailed dose-response / event-study tables
            are available via ``to_dataframe(level=...)``.
        """
        return {
            "att": self.att,
            "se": self.se,
            "t_stat": self.t_stat,
            "p_value": self.p_value,
            "conf_int_lower": self.overall_att_conf_int[0],
            "conf_int_upper": self.overall_att_conf_int[1],
            "acrt": self.overall_acrt,
            "acrt_se": self.overall_acrt_se,
            "acrt_t_stat": self.overall_acrt_t_stat,
            "acrt_p_value": self.overall_acrt_p_value,
            "acrt_conf_int_lower": self.overall_acrt_conf_int[0],
            "acrt_conf_int_upper": self.overall_acrt_conf_int[1],
            "n_obs": self.n_obs,
            "n_treated_units": self.n_treated_units,
            "n_control_units": self.n_control_units,
            "control_group": self.control_group,
            "treatment_type": self.treatment_type,
            "estimation_method": self.estimation_method,
            "degree": self.degree,
            "num_knots": self.num_knots,
            "base_period": self.base_period,
            "anticipation": self.anticipation,
            "n_bootstrap": self.n_bootstrap,
            "alpha": self.alpha,
        }

    def to_dataframe(self, level: str = "dose_response") -> pd.DataFrame:
        """
        Convert results to DataFrame.

        Parameters
        ----------
        level : str, default="dose_response"
            ``"dose_response"``, ``"group_time"``, or ``"event_study"``.
        """
        if level == "dose_response":
            att_df = self.dose_response_att.to_dataframe()
            acrt_df = self.dose_response_acrt.to_dataframe()
            return pd.DataFrame(
                {
                    "dose": att_df["dose"],
                    "att": att_df["effect"],
                    "att_se": att_df["se"],
                    "att_ci_lower": att_df["conf_int_lower"],
                    "att_ci_upper": att_df["conf_int_upper"],
                    "acrt": acrt_df["effect"],
                    "acrt_se": acrt_df["se"],
                    "acrt_ci_lower": acrt_df["conf_int_lower"],
                    "acrt_ci_upper": acrt_df["conf_int_upper"],
                }
            )
        elif level == "group_time":
            rows = []
            for (g, t), data in sorted(self.group_time_effects.items()):
                rows.append(
                    {
                        "group": g,
                        "time": t,
                        "att_glob": data.get("att_glob", np.nan),
                        "acrt_glob": data.get("acrt_glob", np.nan),
                        "n_treated": data.get("n_treated", 0),
                        "n_control": data.get("n_control", 0),
                    }
                )
            return pd.DataFrame(rows)
        elif level == "event_study":
            if self.event_study_effects is None:
                raise ValueError(
                    "Event study effects not computed. Call "
                    "results.aggregate('event_study') for the unified "
                    "post-fit container (on a bootstrapped fit, re-fit "
                    "with n_bootstrap=0 or use the deprecated fit-time "
                    "aggregate='eventstudy'); a result unpickled from an "
                    "older release must be re-fit with diff-diff >= 3.9."
                )
            rows = []
            for rel_t, data in sorted(self.event_study_effects.items()):
                rows.append(
                    {
                        "relative_period": rel_t,
                        "att_glob": data["effect"],
                        "se": data["se"],
                        "t_stat": data["t_stat"],
                        "p_value": data["p_value"],
                        "conf_int_lower": data["conf_int"][0],
                        "conf_int_upper": data["conf_int"][1],
                    }
                )
            return pd.DataFrame(rows)
        else:
            raise ValueError(
                f"Unknown level: {level}. Use 'dose_response', 'group_time', or 'event_study'."
            )

    @property
    def is_significant(self) -> bool:
        """Check if overall ATT is significant."""
        return bool(self.overall_att_p_value < self.alpha)

    @property
    def significance_stars(self) -> str:
        """Significance stars for overall ATT."""
        return _get_significance_stars(self.overall_att_p_value)

    # ------------------------------------------------------------------
    # Post-fit aggregation (row M-025, on the M-122 contract).
    # MIXED architecture: 'simple' and 'dose' are pure VIEWS over stored
    # public fields (the dCDH precedent - nothing recomputed, so they
    # work on ANY fit including bootstrap fits and legacy pickles,
    # relaying the stored inference verbatim); 'event_study' is a KIT
    # RECOMPUTE (the EfficientDiD class) from the pruned per-cell IF
    # payload, failing closed on bootstrap fits.
    # ------------------------------------------------------------------

    def _stored_inference_df(self) -> float:
        """The df the STORED overall inference actually used (NaN = none).

        Bootstrap fits carry percentile p/CI - no df governs them.
        Otherwise ``dose_response_att.df_survey`` is the stored provenance
        channel for fit's ``_survey_df`` (the same value every
        ``safe_inference`` call received); the replicate-undefined
        0-sentinel and ``None`` both report NaN in the df COLUMN, while
        the view relays still pass the RAW stored value into their own
        ``safe_inference`` derivations (``DoseResponseCurve.to_dataframe``
        parity).
        """
        if self.n_bootstrap > 0:
            return float("nan")
        df_survey = self.dose_response_att.df_survey
        if df_survey is not None and np.isfinite(df_survey) and df_survey > 0:
            return float(df_survey)
        return float("nan")

    def _aggregate_compute(
        self, level: str, *, weights: Optional[str], balance_e: Optional[int]
    ) -> Any:
        if level == "simple":
            # 2-row VIEW of the stored overall estimands: ContinuousDiD's
            # headline parameters are the binarized overall ATT (ATT^{loc}
            # under PT; equals ATT^{glob} under SPT) AND ACRT^{glob}, so the
            # target column discriminates two "overall" rows (the
            # container spec's dual-estimand case). Relays are strictly
            # bit-exact - on bootstrap fits the stored quintet includes a
            # FINITE safe_inference t beside the percentile p/CI and it
            # relays through unchanged; only the df column is NaN there.
            att_ci = self.overall_att_conf_int
            acrt_ci = self.overall_acrt_conf_int
            n_total = float(self.n_treated_units + self.n_control_units)
            df_val = self._stored_inference_df()
            return AggregationResult(
                level="simple",
                label=np.array(["overall", "overall"], dtype=object),
                target=np.array(["att", "acrt"], dtype=object),
                att=np.array([self.overall_att, self.overall_acrt], dtype=float),
                se=np.array([self.overall_att_se, self.overall_acrt_se], dtype=float),
                t_stat=np.array([self.overall_att_t_stat, self.overall_acrt_t_stat], dtype=float),
                p_value=np.array(
                    [self.overall_att_p_value, self.overall_acrt_p_value],
                    dtype=float,
                ),
                conf_int_lower=np.array([att_ci[0], acrt_ci[0]], dtype=float),
                conf_int_upper=np.array([att_ci[1], acrt_ci[1]], dtype=float),
                # Treated and control unit sets are DISJOINT for this
                # estimator (unlike Imputation/TwoStage), so the CS
                # disjoint-total convention applies.
                n=np.array([n_total, n_total], dtype=float),
                df=np.array([df_val, df_val], dtype=float),
                alpha=self.alpha,
                n_kind="units",
                weight=np.array([1.0, 1.0], dtype=float),
                estimator="ContinuousDiD",
            )

        if level == "dose":
            # 2N-row VIEW of the stored dose-response curves: att block
            # then acrt block (first-appearance target order). t/p
            # reproduce each DoseResponseCurve.to_dataframe exactly -
            # including the bootstrap branch (stored p, NaN t) and the
            # raw stored df_survey (0-sentinel included) fed to
            # safe_inference on the analytical branch.
            blocks = []
            for curve, target in (
                (self.dose_response_att, "att"),
                (self.dose_response_acrt, "acrt"),
            ):
                n_grid = len(curve.effects)
                if curve.n_bootstrap > 0 and curve.p_value is not None:
                    t_stat = np.full(n_grid, np.nan)
                    p_value = np.asarray(curve.p_value, dtype=float)
                else:
                    t_stat = np.full(n_grid, np.nan)
                    p_value = np.full(n_grid, np.nan)
                    for i in range(n_grid):
                        t_i, p_i, _ = safe_inference(
                            curve.effects[i], curve.se[i], df=curve.df_survey
                        )
                        t_stat[i] = t_i
                        p_value[i] = p_i
                blocks.append((curve, target, t_stat, p_value))
            df_val = self._stored_inference_df()
            return AggregationResult(
                level="dose",
                label=np.concatenate(
                    [np.asarray(c.dose_grid, dtype=object) for c, _, _, _ in blocks]
                ),
                target=np.array(
                    ["att"] * len(blocks[0][0].effects) + ["acrt"] * len(blocks[1][0].effects),
                    dtype=object,
                ),
                att=np.concatenate([c.effects for c, _, _, _ in blocks]).astype(float),
                se=np.concatenate([c.se for c, _, _, _ in blocks]).astype(float),
                t_stat=np.concatenate([t for _, _, t, _ in blocks]),
                p_value=np.concatenate([p for _, _, _, p in blocks]),
                conf_int_lower=np.concatenate([c.conf_int_lower for c, _, _, _ in blocks]).astype(
                    float
                ),
                conf_int_upper=np.concatenate([c.conf_int_upper for c, _, _, _ in blocks]).astype(
                    float
                ),
                # Grid evaluation points carry no count and no aggregation
                # mass - inventing either would be a fabricated number.
                n=np.full(2 * len(blocks[0][0].effects), np.nan),
                df=np.full(2 * len(blocks[0][0].effects), df_val),
                alpha=self.alpha,
                n_kind=None,
                weight=None,
                estimator="ContinuousDiD",
            )

        # level == "event_study": the kit recompute.
        kit = self._aggregation_kit
        if kit is None:
            raise ValueError(
                "This ContinuousDiDResults has no aggregation kit - it is "
                "attached by ContinuousDiD.fit(); a result unpickled from "
                "an older release will not have one. Re-fit with "
                "diff-diff >= 3.9 to enable post-fit aggregate()."
            )
        bk = kit.bookkeeping
        if bk["n_bootstrap"] > 0:
            raise NotImplementedError(
                "aggregate('event_study') on a bootstrapped ContinuousDiD "
                "fit is not implemented - the fit-time event study used "
                "multiplier-bootstrap inference whose per-cell draws are "
                "not retained, and an analytical recompute would silently "
                "differ. Until 4.0 the deprecated fit-time "
                "aggregate='eventstudy' still computes the bootstrap "
                "surface, or re-fit with n_bootstrap=0 for the analytical "
                "post-fit route."
            )
        host = _ContinuousKitAggregator(alpha=kit.alpha)
        es = host._aggregate_event_study(
            bk["gt_summary"],
            gt_bootstrap_info=None,
            unit_survey_weights=bk["unit_survey_weights"],
            unit_cohorts=bk["unit_cohorts"],
            anticipation=kit.anticipation,
        )
        if bk["has_post_cells"]:
            host._compute_event_study_inference(
                es,
                gt_summary=bk["gt_summary"],
                gt_es_payload=bk["gt_es_payload"],
                n_units=bk["n_units"],
                unit_cohorts=bk["unit_cohorts"],
                unit_survey_weights=bk["unit_survey_weights"],
                unit_first_panel_row=bk["unit_first_panel_row"],
                resolved_survey=bk["resolved_survey"],
                survey_df=bk["survey_df"],
            )
        # else: fit-faithful empty-post_gt quirk - the fit-time surface
        # also leaves ES rows at NaN inference when no post-treatment
        # cells exist.
        meta = bk["survey_metadata"]
        meta = dataclasses.replace(meta) if meta is not None else None
        # Per-row df provenance: the kit's survey df is the value this
        # route's safe_inference calls received; 0-sentinel normalized.
        _es_df = bk["survey_df"]
        carrier = dataclasses.replace(
            self,
            event_study_effects=es,
            survey_metadata=meta,
            alpha=kit.alpha,
            anticipation=kit.anticipation,
            event_study_df=(float(_es_df) if _es_df is not None and _es_df > 0 else None),
            # _provenance_kwargs reads base_period off the carrier - it
            # rides the kit like its siblings alpha/anticipation so
            # post-fit mutation of the public field cannot reach
            # recomputed provenance.
            base_period=bk["base_period"],
        )
        return build_event_study_surface(carrier)
