"""
Result containers for the Stacked DiD estimator.

This module contains StackedDiDResults dataclass for Wing, Freedman &
Hollingsworth (2024) stacked difference-in-differences estimation.
"""

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from diff_diff._deprecation import deprecated_field_property
from diff_diff.aggregation import AggregationMixin, AggregationResult
from diff_diff.results import _format_survey_block, _get_significance_stars
from diff_diff.results_base import BaseResults, build_event_study_surface

__all__ = [
    "StackedDiDResults",
]


@dataclass
class StackedDiDResults(BaseResults, AggregationMixin):
    """
    Results from Stacked DiD estimation (Wing, Freedman & Hollingsworth 2024).

    Attributes
    ----------
    overall_att : float
        Overall average treatment effect on the treated (average of
        post-treatment event-study coefficients).
    overall_se : float
        Standard error of overall ATT (delta method on VCV).
    overall_t_stat : float
        T-statistic for overall ATT.
    overall_p_value : float
        P-value for overall ATT.
    overall_conf_int : tuple
        Confidence interval for overall ATT.
    event_study_effects : dict, optional
        Dictionary mapping event time h to effect dict with keys:
        'effect', 'se', 't_stat', 'p_value', 'conf_int', 'n_obs'.
        Always populated on 3.9+ fits (the pooled regression always
        includes the event-time interactions and the surface is always
        extracted - row M-024); None only on pre-3.9 pickles. View it as
        the unified container via ``aggregate('event_study')``.
    group_effects : dict, optional
        Dictionary mapping cohort g to effect dict.
    stacked_data : pd.DataFrame
        Full stacked dataset with _sub_exp, _event_time, _D_sa,
        _Q_weight columns. Accessible for custom analysis.
    groups : list
        Adoption events in the trimmed set (Omega_kappa).
    trimmed_groups : list
        Adoption events excluded by IC1/IC2.
    time_periods : list
        All time periods in the original data.
    n_obs : int
        Number of observations in the original data.
    n_stacked_obs : int
        Number of observations in the stacked dataset.
    n_sub_experiments : int
        Number of sub-experiments in the stack.
    n_treated_units : int
        Distinct treated units across trimmed set.
    n_control_units : int
        Distinct control units across trimmed set.
    kappa_pre : int
        Pre-treatment event-time window size.
    kappa_post : int
        Post-treatment event-time window size.
    weighting : str
        Weighting scheme used.
    control_group : str
        Control-group (clean-control) definition used. (The deprecated
        read-only alias ``clean_control`` warns and returns this value;
        removed in 4.0 - row M-095.)
    alpha : float
        Significance level used.
    event_study_vcov : np.ndarray, optional
        Full event-study variance-covariance matrix: the sub-block of the
        pooled stacked-regression coefficient covariance over the estimated
        ``D_sa x event-time`` interaction columns, ordered by
        ``event_study_vcov_index``. The reported per-event-time SEs are
        exactly ``sqrt(diag())`` of this matrix in every inference mode
        (analytical hc1/hc2_bm sandwich, survey replicate refit, and survey
        TSL all produce the coefficient covariance the SEs are read from).
        The reference period is synthesized, never a regression column, so
        it is absent from the index. Always populated on 3.9+ fits (the
        event-study surface is always materialized - row M-024); None
        only on pre-3.9 pickles.
    event_study_vcov_index : list of int, optional
        Event-time labels ordering ``event_study_vcov``'s rows/columns
        (the estimated event times, reference excluded).
    event_study_df : dict, optional
        Per-event-time inference degrees of freedom PROVENANCE: maps each
        estimated event time to the df actually passed to
        ``safe_inference`` for its stored p-value/CI (per-event
        Bell-McCaffrey Satterthwaite df under ``hc2_bm``; the scalar survey
        df under survey designs; the ``df_convention``-resolved analytical
        fallback otherwise — finite residual df under the 3.9 default,
        ``G − 1`` under "cluster"), or NaN when the row used normal theory
        (``df_convention="normal"``), the df was undefined, or hc2_bm
        failed closed. Always populated on 3.9+ fits (row M-024); None
        only on pre-3.9 pickles.
    df_convention : str, optional
        The estimator's ``df_convention`` configuration echoed onto the
        results ("residual" | "cluster" | "normal"; added 3.9).
    inference_df : float, optional
        The df the stored overall-ATT p-value/CI's ``safe_inference``
        actually received: the BM contrast df under ``hc2_bm``, the
        survey/replicate df on survey fits, else the
        ``df_convention``-resolved analytical fallback. None when the
        overall inference used normal theory or failed closed.
    """

    overall_att: float
    overall_se: float
    overall_t_stat: float
    overall_p_value: float
    overall_conf_int: Tuple[float, float]
    event_study_effects: Optional[Dict[int, Dict[str, Any]]]
    group_effects: Optional[Dict[Any, Dict[str, Any]]]
    stacked_data: pd.DataFrame = field(repr=False)
    groups: List[Any] = field(default_factory=list)
    trimmed_groups: List[Any] = field(default_factory=list)
    time_periods: List[Any] = field(default_factory=list)
    n_obs: int = 0
    n_stacked_obs: int = 0
    n_sub_experiments: int = 0
    n_treated_units: int = 0
    n_control_units: int = 0
    kappa_pre: int = 1
    kappa_post: int = 1
    weighting: str = "aggregate"
    control_group: str = "not_yet_treated"
    alpha: float = 0.05
    anticipation: int = 0
    # Analytical variance family configured at fit time (Phase 1b 2/8). When
    # survey_design= is supplied the survey TSL/replicate variance overrides
    # the analytical family; this field still records the configured value.
    vcov_type: str = "hc1"
    # Cluster identity ("unit" or "unit_subexp") and realized cluster count
    # at fit time. Used by summary() to render the correct CR1/CR2-BM label
    # via `_format_vcov_label(cluster_name=, n_clusters=)`. Per CI codex R2
    # P2: passing cluster_name=None mislabelled clustered StackedDiD fits
    # as one-way HC1/HC2-BM. StackedDiD is intrinsically clustered.
    cluster_name: Optional[str] = None
    n_clusters: Optional[int] = None
    # Survey design metadata (SurveyMetadata instance from diff_diff.survey)
    survey_metadata: Optional[Any] = field(default=None)
    # --- Covariate balancing (CBWSDID, Ustyuzhanin 2026) ---
    # balance: "none" (default, plain weighted stacked DID) or "entropy". When
    # "entropy", `covariates` lists the balanced columns and `balance_diagnostics`
    # maps each sub-experiment a to {n_treated, n_control, effective_control_mass
    # (Ñ^C_a), ess, max_imbalance_pre, max_imbalance_post, balance_solver}. When
    # balancing, `stacked_data` carries `_b_sa` (raw design weights) and the
    # `_Q_weight` column holds the composed final weights W_sa.
    balance: str = "none"
    covariates: Optional[List[str]] = None
    balance_diagnostics: Optional[Dict[Any, Dict[str, Any]]] = field(default=None)
    # Unified event-study surface support (spec section 5, row M-092): the
    # full ES VCV sub-block + ordered horizon index + per-event df actually
    # used. See the class docstring for semantics.
    event_study_vcov: Optional[np.ndarray] = field(default=None, repr=False)
    event_study_vcov_index: Optional[List[int]] = field(default=None, repr=False)
    event_study_df: Optional[Dict[int, float]] = field(default=None, repr=False)
    # Appended LAST (generated __init__ positional indexes are public API).
    df_convention: Optional[str] = None
    inference_df: Optional[float] = None

    # Deprecated read-only alias for ``control_group`` (row M-095; removed
    # in 4.0). No annotation, so it stays a descriptor and never becomes a
    # __dataclass_fields__ entry.
    clean_control = deprecated_field_property("StackedDiDResults", "clean_control", "control_group")

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Migrate pickles created before the ``clean_control`` ->
        ``control_group`` rename (row M-095): rewrite the key on load so
        both the new field and the deprecated alias work on old pickles."""
        if "clean_control" in state and "control_group" not in state:
            state = dict(state)
            state["control_group"] = state.pop("clean_control")
        self.__dict__.update(state)

    # ------------------------------------------------------------------
    # Container-consumer provenance (rows M-024 / M-093). Class-level
    # attrs, deliberately NOT dataclass fields: the generated __init__'s
    # positional indexes are public API, and both values are derivable.
    # ------------------------------------------------------------------
    #: Every sub-experiment normalizes against the single omitted
    #: reference ``e = -1 - anticipation`` - universal-base semantics in
    #: the CallawaySantAnna vocabulary. Read by ``_provenance_kwargs``
    #: so honest_did's cannot-verify-universal-base fail-safe stays
    #: silent on StackedDiD containers.
    base_period: ClassVar[str] = "universal"

    @property
    def reference_event_times(self) -> Tuple[int, ...]:
        """The singleton common reference event time, ``(-1 - anticipation,)``.

        StackedDiD has exactly one omitted reference shared by every
        sub-experiment, so the container consumers' common-reference
        guard always sees a single entry (rows M-024 / M-093).
        """
        return (-1 - int(self.anticipation),)

    # ------------------------------------------------------------------
    # Post-fit aggregation (row M-024, on the M-122 contract). Both
    # levels are pure VIEWS over stored fields - the event-study surface
    # is always materialized at fit since 3.9, and "simple" relays the
    # stored overall inference bit-exactly. Nothing is recomputed, so
    # every inference mode (survey TSL, replicate refit, hc2_bm) relays
    # faithfully.
    # ------------------------------------------------------------------
    # ClassVar: on a dataclass a bare annotation would turn this routing
    # configuration into an ``__init__`` field.
    _AGGREGATE_SUPPORTED: ClassVar[Tuple[str, ...]] = ("simple", "event_study")
    # StackedDiD has no balance_e machinery on any aggregation level
    # (kappa trimming already balances every retained cohort's window).
    _AGGREGATE_BALANCE_E_TYPES: ClassVar[Tuple[str, ...]] = ()

    def _aggregate_compute(
        self, level: str, *, weights: Optional[str], balance_e: Optional[int]
    ) -> Any:
        if level == "event_study":
            # The unified container over ``event_study_effects`` (always
            # populated on 3.9+ fits; pre-3.9 pickles raise the absent-
            # surface error with a re-fit hint).
            return build_event_study_surface(self)

        # level == "simple": one-row view relaying the stored overall
        # inference. ``n`` is the TREATED-unit count: StackedDiD's treated
        # and control sets OVERLAP (a later-treated unit is treated in its
        # own sub-experiment and a clean control in earlier ones), so a
        # disjoint total does not exist as a stored scalar and summing the
        # two counts would double-count - deliberately narrower in scope
        # than CallawaySantAnna's treated+control "units" (REGISTRY
        # StackedDiD M-024 Note; cross-container ``n`` comparisons are out
        # of contract for this estimator). ``target`` is "att" per the CS
        # precedent: ``overall_att`` is the equally-weighted average of
        # post-treatment event-study coefficients, NOT the per-event-time
        # trimmed aggregate ATT, so weighting-specific target strings
        # would misstate the scalar (``describe_target_parameter`` is the
        # estimand's prose source of truth).
        ci = self.overall_conf_int if self.overall_conf_int is not None else (np.nan, np.nan)
        return AggregationResult(
            level="simple",
            label=np.array(["overall"], dtype=object),
            target=np.array(["att"], dtype=object),
            att=np.array([self.overall_att], dtype=float),
            se=np.array([self.overall_se], dtype=float),
            t_stat=np.array([self.overall_t_stat], dtype=float),
            p_value=np.array([self.overall_p_value], dtype=float),
            conf_int_lower=np.array([ci[0]], dtype=float),
            conf_int_upper=np.array([ci[1]], dtype=float),
            n=np.array([float(self.n_treated_units)], dtype=float),
            df=np.array(
                [float(self.inference_df) if self.inference_df is not None else float("nan")],
                dtype=float,
            ),
            alpha=self.alpha,
            n_kind="units",
            weight=np.array([1.0], dtype=float),
            estimator=type(self).__name__.replace("Results", ""),
        )

    # --- Inference-field aliases (balance/external-adapter compatibility) ---
    @property
    def att(self) -> float:
        return self.overall_att

    @property
    def se(self) -> float:
        return self.overall_se

    @property
    def conf_int(self) -> Tuple[float, float]:
        return self.overall_conf_int

    @property
    def p_value(self) -> float:
        return self.overall_p_value

    @property
    def t_stat(self) -> float:
        return self.overall_t_stat

    def __repr__(self) -> str:
        """Concise string representation."""
        sig = _get_significance_stars(self.overall_p_value)
        return (
            f"StackedDiDResults(ATT={self.overall_att:.4f}{sig}, "
            f"SE={self.overall_se:.4f}, "
            f"n_sub_exp={self.n_sub_experiments}, "
            f"n_stacked_obs={self.n_stacked_obs})"
        )

    @property
    def coef_var(self) -> float:
        """Coefficient of variation: SE / abs(overall ATT). NaN when ATT is 0 or SE non-finite."""
        if not (np.isfinite(self.overall_se) and self.overall_se >= 0):
            return np.nan
        if not np.isfinite(self.overall_att) or self.overall_att == 0:
            return np.nan
        return self.overall_se / abs(self.overall_att)

    def summary(self, alpha: Optional[float] = None) -> str:
        """
        Generate formatted summary of estimation results.

        Parameters
        ----------
        alpha : float, optional
            Significance level. Defaults to alpha used in estimation.

        Returns
        -------
        str
            Formatted summary.
        """
        alpha = alpha or self.alpha
        conf_level = int((1 - alpha) * 100)

        lines = [
            "=" * 85,
            "Stacked DiD Estimator Results (Wing, Freedman & Hollingsworth 2024)".center(85),
            "=" * 85,
            "",
            f"{'Original observations:':<30} {self.n_obs:>10}",
            f"{'Stacked observations:':<30} {self.n_stacked_obs:>10}",
            f"{'Sub-experiments:':<30} {self.n_sub_experiments:>10}",
            f"{'Treated units:':<30} {self.n_treated_units:>10}",
            f"{'Control units:':<30} {self.n_control_units:>10}",
            f"{'Treatment cohorts:':<30} {len(self.groups):>10}",
            f"{'Trimmed cohorts:':<30} {len(self.trimmed_groups):>10}",
            f"{'Event window:':<30} {'[' + str(-self.kappa_pre) + ', ' + str(self.kappa_post) + ']':>10}",
            f"{'Weighting:':<30} {self.weighting:>10}",
            f"{'Control group:':<30} {self.control_group:>10}",
            "",
        ]

        # Variance family label (per CI codex R1 P2): surface the analytical
        # vcov_type when the survey path didn't override. Per R2 P2: pass
        # cluster_name + n_clusters so the label renders as "CR1 cluster-
        # robust at unit, G=N" rather than the one-way "HC1 heteroskedasticity-
        # robust" — StackedDiD is intrinsically clustered.
        if self.survey_metadata is None and self.vcov_type:
            from diff_diff.results import _format_vcov_label

            label = _format_vcov_label(
                self.vcov_type,
                cluster_name=self.cluster_name,
                n_clusters=self.n_clusters,
                n_obs=self.n_stacked_obs,
            )
            if label is not None:
                lines.append(f"{'Variance:':<30} {label:>50}")
                lines.append("")

        # Add survey design info
        if self.survey_metadata is not None:
            sm = self.survey_metadata
            lines.extend(_format_survey_block(sm, 85))

        # Overall ATT
        lines.extend(
            [
                "-" * 85,
                "Overall Average Treatment Effect on the Treated".center(85),
                "-" * 85,
                f"{'Parameter':<15} {'Estimate':>12} {'Std. Err.':>12} "
                f"{'t-stat':>10} {'P>|t|':>10} {'Sig.':>6}",
                "-" * 85,
            ]
        )

        t_str = (
            f"{self.overall_t_stat:>10.3f}" if np.isfinite(self.overall_t_stat) else f"{'NaN':>10}"
        )
        p_str = (
            f"{self.overall_p_value:>10.4f}"
            if np.isfinite(self.overall_p_value)
            else f"{'NaN':>10}"
        )
        sig = _get_significance_stars(self.overall_p_value)

        lines.extend(
            [
                f"{'ATT':<15} {self.overall_att:>12.4f} {self.overall_se:>12.4f} "
                f"{t_str} {p_str} {sig:>6}",
                "-" * 85,
                "",
                f"{conf_level}% Confidence Interval: "
                f"[{self.overall_conf_int[0]:.4f}, {self.overall_conf_int[1]:.4f}]",
            ]
        )

        cv = self.coef_var
        if np.isfinite(cv):
            lines.append(f"{'CV (SE/abs(ATT)):':<25} {cv:>10.4f}")

        lines.append("")

        # Event study effects
        if self.event_study_effects:
            lines.extend(
                [
                    "-" * 85,
                    "Event Study (Dynamic) Effects".center(85),
                    "-" * 85,
                    f"{'Rel. Period':<15} {'Estimate':>12} {'Std. Err.':>12} "
                    f"{'t-stat':>10} {'P>|t|':>10} {'Sig.':>6}",
                    "-" * 85,
                ]
            )

            for h in sorted(self.event_study_effects.keys()):
                eff = self.event_study_effects[h]
                if eff.get("n_obs", 1) == 0:
                    # Reference period marker
                    lines.append(
                        f"[ref: {h}]" f"{'0.0000':>17} {'---':>12} {'---':>10} {'---':>10} {'':>6}"
                    )
                elif np.isnan(eff["effect"]):
                    lines.append(f"{h:<15} {'NaN':>12} {'NaN':>12} {'NaN':>10} {'NaN':>10} {'':>6}")
                else:
                    e_sig = _get_significance_stars(eff["p_value"])
                    e_t = (
                        f"{eff['t_stat']:>10.3f}" if np.isfinite(eff["t_stat"]) else f"{'NaN':>10}"
                    )
                    e_p = (
                        f"{eff['p_value']:>10.4f}"
                        if np.isfinite(eff["p_value"])
                        else f"{'NaN':>10}"
                    )
                    lines.append(
                        f"{h:<15} {eff['effect']:>12.4f} {eff['se']:>12.4f} "
                        f"{e_t} {e_p} {e_sig:>6}"
                    )

            lines.extend(["-" * 85, ""])

        # Group effects
        if self.group_effects:
            lines.extend(
                [
                    "-" * 85,
                    "Group (Cohort) Effects".center(85),
                    "-" * 85,
                    f"{'Cohort':<15} {'Estimate':>12} {'Std. Err.':>12} "
                    f"{'t-stat':>10} {'P>|t|':>10} {'Sig.':>6}",
                    "-" * 85,
                ]
            )

            for g in sorted(self.group_effects.keys()):
                eff = self.group_effects[g]
                if np.isnan(eff["effect"]):
                    lines.append(f"{g:<15} {'NaN':>12} {'NaN':>12} {'NaN':>10} {'NaN':>10} {'':>6}")
                else:
                    g_sig = _get_significance_stars(eff["p_value"])
                    g_t = (
                        f"{eff['t_stat']:>10.3f}" if np.isfinite(eff["t_stat"]) else f"{'NaN':>10}"
                    )
                    g_p = (
                        f"{eff['p_value']:>10.4f}"
                        if np.isfinite(eff["p_value"])
                        else f"{'NaN':>10}"
                    )
                    lines.append(
                        f"{g:<15} {eff['effect']:>12.4f} {eff['se']:>12.4f} "
                        f"{g_t} {g_p} {g_sig:>6}"
                    )

            lines.extend(["-" * 85, ""])

        lines.extend(
            [
                "Signif. codes: '***' 0.001, '**' 0.01, '*' 0.05, '.' 0.1",
                "=" * 85,
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
            Canonical inference row plus scalar metadata. Detailed
            event-study / group tables are available via
            ``to_dataframe(level=...)``.
        """
        result = {
            "att": self.att,
            "se": self.se,
            "t_stat": self.t_stat,
            "p_value": self.p_value,
            "conf_int_lower": self.overall_conf_int[0],
            "conf_int_upper": self.overall_conf_int[1],
            "n_obs": self.n_obs,
            "n_stacked_obs": self.n_stacked_obs,
            "n_sub_experiments": self.n_sub_experiments,
            "n_treated_units": self.n_treated_units,
            "n_control_units": self.n_control_units,
            "kappa_pre": self.kappa_pre,
            "kappa_post": self.kappa_post,
            "weighting": self.weighting,
            "control_group": self.control_group,
            # Deprecated key mirroring ``control_group`` through the 3.9
            # shim window; dropped in 4.0 (row M-095, section 5 policy).
            "clean_control": self.control_group,
            "anticipation": self.anticipation,
            "alpha": self.alpha,
            "vcov_type": self.vcov_type,
        }
        if self.cluster_name is not None:
            result["cluster_name"] = self.cluster_name
        if self.n_clusters is not None:
            result["n_clusters"] = self.n_clusters
        if self.df_convention is not None:
            result["df_convention"] = self.df_convention
        if self.inference_df is not None:
            result["inference_df"] = self.inference_df
        return result

    def to_dataframe(self, level: str = "event_study") -> pd.DataFrame:
        """
        Convert results to DataFrame.

        Parameters
        ----------
        level : str, default="event_study"
            Level of aggregation:
            - "event_study": Event study effects by relative time
            - "group": Group (cohort) effects

        Returns
        -------
        pd.DataFrame
            Results as DataFrame.
        """
        if level == "event_study":
            if self.event_study_effects is None:
                # Only reachable on pre-3.9 pickles: 3.9+ fits always
                # materialize the surface (row M-024).
                raise ValueError(
                    "Event study effects not present on this results object. "
                    "Re-fit with diff-diff >= 3.9, which always computes the "
                    "event-study surface."
                )
            rows = []
            for h, data in sorted(self.event_study_effects.items()):
                rows.append(
                    {
                        "relative_period": h,
                        "effect": data["effect"],
                        "se": data["se"],
                        "t_stat": data["t_stat"],
                        "p_value": data["p_value"],
                        "conf_int_lower": data["conf_int"][0],
                        "conf_int_upper": data["conf_int"][1],
                        "n_obs": data.get("n_obs", np.nan),
                    }
                )
            return pd.DataFrame(rows)

        elif level == "group":
            raise ValueError(
                "Group aggregation is not supported by StackedDiD. "
                "The pooled stacked regression cannot produce cohort-specific "
                "effects. Use CallawaySantAnna or ImputationDiD for "
                "cohort-level estimates."
            )

        else:
            raise ValueError(f"Unknown level: {level}. Use 'event_study' or 'group'.")

    @property
    def is_significant(self) -> bool:
        """Check if overall ATT is significant."""
        return bool(self.overall_p_value < self.alpha)

    @property
    def significance_stars(self) -> str:
        """Significance stars for overall ATT."""
        return _get_significance_stars(self.overall_p_value)
