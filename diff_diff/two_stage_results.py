"""
Result containers for the Two-Stage DiD estimator.

This module contains TwoStageBootstrapResults and TwoStageDiDResults
dataclasses. Extracted from two_stage.py for module size management.
"""

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from diff_diff.aggregation import AggregationMixin, AggregationResult, build_total_relay_row
from diff_diff.results import _format_survey_block, _get_significance_stars
from diff_diff.results_base import BaseResults, build_event_study_surface
from diff_diff.two_stage_aggregation import _TwoStageAggregationMixin


class _TwoStageKitAggregator(_TwoStageAggregationMixin):
    """Throwaway per-call host for the post-fit recompute (M-022/M-119).

    Hosts the moved Stage-2/GMM methods with exactly the mixin's declared
    host-attribute contract, populated from KIT SNAPSHOTS — never from the
    live estimator (whose config may have been mutated since the fit) and
    never from mutable public results fields. A fresh instance per
    ``aggregate()`` call; the moved methods write nothing to ``self``, so
    the kit stays immutable either way.
    """

    def __init__(
        self,
        *,
        alpha: float,
        pretrends: bool,
        horizon_max: Optional[int],
        rank_deficient_action: str,
    ) -> None:
        self.alpha = alpha
        self.pretrends = pretrends
        self.horizon_max = horizon_max
        self.rank_deficient_action = rank_deficient_action


__all__ = [
    "TwoStageBootstrapResults",
    "TwoStageDiDResults",
]


@dataclass
class TwoStageBootstrapResults:
    """
    Results from TwoStageDiD bootstrap inference.

    Bootstrap uses multiplier bootstrap on the GMM influence function,
    consistent with other library estimators. The R `did2s` package defaults
    to analytical corrected clustered SEs (``bootstrap = FALSE``); its optional
    block bootstrap (``bootstrap = TRUE``) and this multiplier bootstrap are
    asymptotically equivalent.

    Attributes
    ----------
    n_bootstrap : int
        Number of bootstrap iterations.
    weight_type : str
        Type of bootstrap weights: "rademacher", "mammen", or "webb".
    alpha : float
        Significance level used for confidence intervals.
    overall_att_se : float
        Bootstrap standard error for overall ATT.
    overall_att_ci : tuple
        Bootstrap confidence interval for overall ATT.
    overall_att_p_value : float
        Bootstrap p-value for overall ATT.
    event_study_ses : dict, optional
        Bootstrap SEs for event study effects.
    event_study_cis : dict, optional
        Bootstrap CIs for event study effects.
    event_study_p_values : dict, optional
        Bootstrap p-values for event study effects.
    group_ses : dict, optional
        Bootstrap SEs for group effects.
    group_cis : dict, optional
        Bootstrap CIs for group effects.
    group_p_values : dict, optional
        Bootstrap p-values for group effects.
    bootstrap_distribution : np.ndarray, optional
        Full bootstrap distribution of overall ATT.
    """

    n_bootstrap: int
    weight_type: str
    alpha: float
    overall_att_se: float
    overall_att_ci: Tuple[float, float]
    overall_att_p_value: float
    event_study_ses: Optional[Dict[int, float]] = None
    event_study_cis: Optional[Dict[int, Tuple[float, float]]] = None
    event_study_p_values: Optional[Dict[int, float]] = None
    group_ses: Optional[Dict[Any, float]] = None
    group_cis: Optional[Dict[Any, Tuple[float, float]]] = None
    group_p_values: Optional[Dict[Any, float]] = None
    bootstrap_distribution: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class TwoStageDiDResults(BaseResults, AggregationMixin):
    """
    Results from Gardner (2022) two-stage DiD estimation.

    Attributes
    ----------
    treatment_effects : pd.DataFrame
        Per-observation treatment effects with columns: unit, time,
        tau_hat, weight. tau_hat is the residualized outcome y_tilde
        for treated observations; weight is 1/n_treated.
    overall_att : float
        Overall average treatment effect on the treated.
    overall_se : float
        Standard error of overall ATT (GMM sandwich).
    overall_t_stat : float
        T-statistic for overall ATT.
    overall_p_value : float
        P-value for overall ATT.
    overall_conf_int : tuple
        Confidence interval for overall ATT.
    event_study_effects : dict, optional
        Dictionary mapping relative time h to effect dict with keys:
        'effect', 'se', 't_stat', 'p_value', 'conf_int', 'n_obs'.
    group_effects : dict, optional
        Dictionary mapping cohort g to effect dict.
    groups : list
        List of treatment cohorts.
    time_periods : list
        List of all time periods.
    n_obs : int
        Total number of observations.
    n_treated_obs : int
        Number of treated observations.
    n_untreated_obs : int
        Number of untreated observations.
    n_treated_units : int
        Number of ever-treated units.
    n_control_units : int
        Number of units contributing to untreated observations.
    alpha : float
        Significance level used.
    bootstrap_results : TwoStageBootstrapResults, optional
        Bootstrap inference results.
    """

    treatment_effects: pd.DataFrame
    overall_att: float
    overall_se: float
    overall_t_stat: float
    overall_p_value: float
    overall_conf_int: Tuple[float, float]
    event_study_effects: Optional[Dict[int, Dict[str, Any]]]
    group_effects: Optional[Dict[Any, Dict[str, Any]]]
    groups: List[Any]
    time_periods: List[Any]
    n_obs: int
    n_treated_obs: int
    n_untreated_obs: int
    n_treated_units: int
    n_control_units: int
    alpha: float = 0.05
    anticipation: int = 0
    bootstrap_results: Optional[TwoStageBootstrapResults] = field(default=None, repr=False)
    # Survey design metadata (SurveyMetadata instance from diff_diff.survey)
    survey_metadata: Optional[Any] = field(default=None, repr=False)
    # --- Variance-estimator metadata (Phase 1b vcov_type threading) ---
    # vcov_type is permanently narrow to {"hc1"} per the Gardner (2022) GMM
    # cluster-sandwich. cluster_name/n_clusters carry the cluster label (the
    # Gardner sandwich always clusters — default at the unit column, see
    # two_stage.py:1547) so summary() renders the unit-cluster CR1 label rather
    # than generic HC1. Both are None under any survey design (the survey block
    # already reports the design's PSU/strata metadata).
    vcov_type: str = "hc1"
    cluster_name: Optional[str] = None
    n_clusters: Optional[int] = None
    # --- Unified event-study surface support (spec section 5, row M-092) ---
    # event_study_vcov: the full Gardner-GMM covariance over the ESTIMATED
    # horizon coefficients, ordered by event_study_vcov_index. The reference
    # period and Proposition-5 horizons are never Stage-2 regression columns,
    # so they appear in event_study_effects but not here; all-filtered
    # horizons (n_obs == 0) ARE columns, carrying the rank guard's NaN
    # rows/columns (consistent with their NaN marginal SEs). Both are None
    # under bootstrap (percentile inference, no covariance) and under
    # replicate-weight survey designs (the replicate VCV has a mixed
    # [overall, ES, groups] layout; not persisted - CS precedent).
    # event_study_df: the scalar df every estimated ES row's safe_inference
    # received (the survey df; None on non-survey fits -> normal theory, and
    # None under bootstrap where the stored inference never used a df).
    event_study_vcov: Optional[np.ndarray] = field(default=None, repr=False)
    event_study_vcov_index: Optional[List[int]] = field(default=None, repr=False)
    event_study_df: Optional[float] = field(default=None, repr=False)
    # Private panel-backed post-fit aggregation kit (rows M-022/M-119),
    # attached by TwoStageDiD.fit(). None on results unpickled from a
    # pre-3.9 release (aggregate() then fails with the re-fit message).
    # Appended LAST (the generated __init__ positional indexes are public
    # API).
    _aggregation_kit: Optional[Any] = field(default=None, repr=False, compare=False)

    # Post-fit aggregation vocabulary (M-022). balance_e keeps the mixin
    # default ("event_study",) - CS precedent, do not redeclare.
    _AGGREGATE_SUPPORTED = ("simple", "event_study", "group", "total")

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

    # --- Post-fit aggregation (M-022/M-119) ------------------------------

    def _aggregate_compute(
        self, level: str, *, weights: Optional[str], balance_e: Optional[int]
    ) -> Any:
        kit = self._aggregation_kit
        if kit is None:
            raise ValueError(
                "This TwoStageDiDResults carries no aggregation kit - it is "
                "attached by TwoStageDiD.fit(), so a result unpickled from "
                "an older release will not have one. Re-fit with "
                "diff-diff >= 3.9 to aggregate post-fit."
            )
        # Per-level bootstrap policy (v4-design section 6, converged with row
        # M-027): 'simple' is a bit-exact RELAY of the stored overall quintet
        # - faithful under any inference regime, bootstrap included - so it
        # dispatches BEFORE the bootstrap gate. Only the RECOMPUTE levels
        # below fail closed on bootstrapped fits. (This supersedes the
        # uniform-conservatism decision recorded with M-022; its rationale is
        # honored by the relay's NaN df column.)
        if level == "simple":
            return self._aggregate_simple_result(kit)
        if level == "total":
            return self._aggregate_total_result(kit)
        if self.bootstrap_results is not None:
            raise NotImplementedError(
                f"aggregate({level!r}) is not yet available on a bootstrapped "
                "fit (n_bootstrap > 0): the per-target bootstrap draws are "
                "not retained, so post-fit re-aggregation cannot replay "
                "percentile inference and analytical inference would "
                "misrepresent the fit. aggregate('simple') and, where "
                "supported, aggregate('total') relay the stored "
                "bootstrap inference and remain available; otherwise re-fit "
                "with the aggregation you need, or use n_bootstrap=0."
            )
        bk = dict(kit.bookkeeping)
        # Fresh throwaway host per call, populated from KIT snapshots only
        # (estimator/config mutation after fit() must not leak in).
        agg = _TwoStageKitAggregator(
            alpha=kit.alpha,
            pretrends=bk["pretrends"],
            horizon_max=bk["horizon_max"],
            rank_deficient_action=bk["rank_deficient_action"],
        )
        common: Dict[str, Any] = dict(
            df=bk["df"],
            unit=bk["unit"],
            time=bk["time"],
            first_treat=bk["first_treat"],
            covariates=bk["covariates"],
            omega_0_mask=bk["omega_0_mask"],
            omega_1_mask=bk["omega_1_mask"],
            unit_fe=bk["unit_fe"],
            time_fe=bk["time_fe"],
            grand_mean=bk["grand_mean"],
            delta_hat=bk["delta_hat"],
            cluster_var=bk["cluster_var"],
            treatment_groups=bk["treatment_groups"],
            kept_cov_mask=bk["kept_cov_mask"],
            survey_weights=bk["survey_weights"],
            survey_weight_type=bk["survey_weight_type"],
            survey_df=bk["survey_df_stage2"],
            resolved_survey=(None if bk["uses_replicate"] else bk["resolved_survey"]),
            score_pad_mask=bk["score_pad_mask"],
            cluster_ids_full=bk["cluster_ids_full"],
        )
        if level == "group":
            effects = agg._stage2_group(**common)
            replay_df_g: Optional[int] = bk["survey_df_stage2"]
            if bk["uses_replicate"]:
                # LEVEL-MATCHED replay: [overall, groups] - reproduces
                # fit(aggregate='group') exactly (see the replay docstring).
                _, _, replay_df_g = agg._replay_replicate_inference(
                    df=bk["df"],
                    outcome=bk["outcome"],
                    unit=bk["unit"],
                    time=bk["time"],
                    first_treat=bk["first_treat"],
                    covariates=bk["covariates"],
                    omega_0_mask=bk["omega_0_mask"],
                    omega_1_mask=bk["omega_1_mask"],
                    cluster_var=bk["cluster_var"],
                    treatment_groups=bk["treatment_groups"],
                    ref_period=bk["ref_period"],
                    balance_e=None,
                    keep_mask=bk["keep_mask"],
                    resolved_survey=bk["resolved_survey"],
                    overall_att=bk["overall_att"],
                    event_study_effects=None,
                    group_effects=effects,
                    survey_df_seed=bk["survey_df_stage2"],
                )
            return self._group_effects_to_aggregation(effects, kit, group_df=replay_df_g)
        # level == "event_study" (the mixin validated the vocabulary)
        es, es_vcov, es_vcov_index = agg._stage2_event_study(
            ref_period=bk["ref_period"], balance_e=balance_e, **common
        )
        replay_df: Optional[int] = None
        if bk["uses_replicate"]:
            _, _, replay_df = agg._replay_replicate_inference(
                df=bk["df"],
                outcome=bk["outcome"],
                unit=bk["unit"],
                time=bk["time"],
                first_treat=bk["first_treat"],
                covariates=bk["covariates"],
                omega_0_mask=bk["omega_0_mask"],
                omega_1_mask=bk["omega_1_mask"],
                cluster_var=bk["cluster_var"],
                treatment_groups=bk["treatment_groups"],
                ref_period=bk["ref_period"],
                balance_e=balance_e,
                keep_mask=bk["keep_mask"],
                resolved_survey=bk["resolved_survey"],
                overall_att=bk["overall_att"],
                event_study_effects=es,
                group_effects=None,
                survey_df_seed=bk["survey_df_stage2"],
            )
        # Carrier + shared builder, reproducing fit's M-092 mode gates
        # exactly (two_stage.py): analytical -> recomputed V + index + the
        # finite-and->0 df scalar; replicate -> vcov/index None with the
        # REPLAYED level-matched df; bootstrap unreachable (failed closed
        # above). The carrier's metadata is a copy-on-use of the KIT's
        # fit-final metadata copy (never the mutable public field).
        if bk["uses_replicate"]:
            vcov_final = None
            index_final = None
            df_gate = replay_df
        else:
            vcov_final = es_vcov
            index_final = es_vcov_index
            df_gate = bk["survey_df_stage2"]
        es_df_final: Optional[float] = (
            float(df_gate) if df_gate is not None and np.isfinite(df_gate) and df_gate > 0 else None
        )
        meta = bk["survey_metadata"]
        if meta is not None:
            if bk["uses_replicate"]:
                meta = dataclasses.replace(
                    meta, df_survey=(replay_df if replay_df and replay_df > 0 else None)
                )
            else:
                meta = dataclasses.replace(meta)
        carrier = dataclasses.replace(
            self,
            event_study_effects=es,
            event_study_vcov=vcov_final,
            event_study_vcov_index=index_final,
            event_study_df=es_df_final,
            survey_metadata=meta,
            anticipation=kit.anticipation,
            alpha=kit.alpha,
        )
        return build_event_study_surface(carrier)

    def _aggregate_simple_result(self, kit: Any) -> AggregationResult:
        """One-row relay of the stored overall inference (bit-exact).

        ``n = n_treated_obs`` (|Omega_1|, the pre-filter D-column count)
        with ``n_kind="obs"``: TwoStageDiD's
        ``n_treated_units``/``n_control_units`` unit sets OVERLAP (an
        eventually-treated unit contributes untreated observations), so
        the CS/EDiD disjoint-units convention cannot apply (the StackedDiD
        carve-out class). The ATT's actual Stage-2 support excludes rows
        whose ``y_tilde`` is non-finite (their treatment indicator is
        zeroed before the OLS), so on such degenerate fits - which warn at
        fit time - ``n`` exceeds the averaged support.

        ``df`` is the kit's ``survey_df_final`` snapshot - the exact value
        the STORED overall ``safe_inference`` received (on a replicate fit
        that value came from the ``[overall]``-only joint stack, which is
        precisely why it must be snapshotted rather than re-derived: a
        post-fit level-matched replay produces a different n_valid).
        None → all-NaN df column; the replicate-undefined 0 sentinel NaNs
        out via post_init. Bootstrapped fits relay the stored quintet
        verbatim with a NaN df column - no df governs percentile
        inference (the per-level policy converged with row M-027).
        """
        return AggregationResult(
            level="simple",
            label=np.array(["overall"], dtype=object),
            target=np.array(["att"], dtype=object),
            att=np.array([self.overall_att], dtype=float),
            se=np.array([self.overall_se], dtype=float),
            t_stat=np.array([self.overall_t_stat], dtype=float),
            p_value=np.array([self.overall_p_value], dtype=float),
            conf_int_lower=np.array([self.overall_conf_int[0]], dtype=float),
            conf_int_upper=np.array([self.overall_conf_int[1]], dtype=float),
            n=np.array([kit.bookkeeping["n_treated_obs"]], dtype=float),
            df=(
                np.nan if self.bootstrap_results is not None else kit.bookkeeping["survey_df_final"]
            ),
            alpha=kit.alpha,
            n_kind="obs",
            weight=np.array([1.0], dtype=float),
            estimator=type(self).__name__.replace("Results", ""),
        )

    def _aggregate_total_result(self, kit: Any) -> AggregationResult:
        """The estimator-owned total incremental outcome as a one-row table.

        Exact relay ``C x overall`` CONDITIONAL on the realized aggregation
        mass, with ``C`` the post-filter treatment-indicator support the
        second stage actually regressed on: ``D = omega_1_mask`` with
        non-finite ``_y_tilde`` rows zeroed (the fit-time masker's inline
        mirror - never re-called post-fit, so its warnings are not
        re-emitted; the kit frame is a private column-subset COPY, so this
        is snapshot-grade). ``score_pad_mask`` is irrelevant here (it pads
        scores/variance only; ``omega_1_mask`` governs D). Degenerate
        ``D.sum() == 0`` maps to a NaN mass -> all-NaN row (mirrors the
        fit's (nan, nan)). Fails closed on fits declaring a
        ``survey_design=`` (see the REGISTRY Note).
        """
        bk = kit.bookkeeping
        if bk["survey_metadata"] is not None:
            raise NotImplementedError(
                "aggregate('total') is not available on fits declaring a "
                "survey_design: the realized-mass relay omits the survey "
                "mass-uncertainty variance term, and design-aware "
                "population-scale totals are not implemented (retained "
                "weight scale differs by design family) - tracked in "
                "DEFERRED.md. For an unweighted clustered fit, cluster= "
                "(without survey_design) supports totals."
            )
        d_mask = np.asarray(bk["omega_1_mask"], dtype=bool)
        y_tilde = np.asarray(bk["df"]["_y_tilde"], dtype=float)
        support = float(np.sum(d_mask & np.isfinite(y_tilde)))
        mass = support if support > 0 else float(np.nan)
        return build_total_relay_row(
            mass=mass,
            att=self.overall_att,
            se=self.overall_se,
            t_stat=self.overall_t_stat,
            p_value=self.overall_p_value,
            conf_int=self.overall_conf_int,
            df=(np.nan if self.bootstrap_results is not None else bk["survey_df_final"]),
            alpha=kit.alpha,
            estimator=type(self).__name__.replace("Results", ""),
        )

    def _group_effects_to_aggregation(
        self, effects: Dict[Any, Dict[str, Any]], kit: Any, *, group_df: Optional[int]
    ) -> AggregationResult:
        """Per-cohort AggregationResult from the recomputed group dict.

        ``df`` is a SCALAR relay (a deliberate divergence from
        ImputationDiD's per-row ``df_used`` capture, documented in the
        REGISTRY note): ``_stage2_group`` passes ONE immutable
        ``survey_df`` parameter to every row's ``safe_inference``, so a
        scalar broadcast is provenance-exact by construction and keeps the
        moved method verbatim. Analytical fits relay the stage-2 seed;
        replicate fits relay the REPLAYED level-matched value (the replay
        rewrote every row's inference under it). ``weight=None``: Stage-2
        cohort dummies carry no cross-cohort mass (the CS rationale).
        ``n_kind="obs"``: ``n_obs`` counts the cohort's treated
        observations backing its indicator column.
        """
        labels = list(effects.keys())
        return AggregationResult(
            level="group",
            label=np.array(labels, dtype=object),
            target=np.array(["att"] * len(labels), dtype=object),
            att=np.array([effects[g]["effect"] for g in labels], dtype=float),
            se=np.array([effects[g]["se"] for g in labels], dtype=float),
            t_stat=np.array([effects[g]["t_stat"] for g in labels], dtype=float),
            p_value=np.array([effects[g]["p_value"] for g in labels], dtype=float),
            conf_int_lower=np.array([effects[g]["conf_int"][0] for g in labels], dtype=float),
            conf_int_upper=np.array([effects[g]["conf_int"][1] for g in labels], dtype=float),
            n=np.array([effects[g]["n_obs"] for g in labels], dtype=float),
            df=(float(group_df) if group_df is not None else None),
            alpha=kit.alpha,
            n_kind="obs",
            weight=None,
            estimator=type(self).__name__.replace("Results", ""),
        )

    def __repr__(self) -> str:
        """Concise string representation."""
        sig = _get_significance_stars(self.overall_p_value)
        return (
            f"TwoStageDiDResults(ATT={self.overall_att:.4f}{sig}, "
            f"SE={self.overall_se:.4f}, "
            f"n_groups={len(self.groups)}, "
            f"n_treated_obs={self.n_treated_obs})"
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
            "Two-Stage DiD Estimator Results (Gardner 2022)".center(85),
            "=" * 85,
            "",
            f"{'Total observations:':<30} {self.n_obs:>10}",
            f"{'Treated observations:':<30} {self.n_treated_obs:>10}",
            f"{'Untreated observations:':<30} {self.n_untreated_obs:>10}",
            f"{'Treated units:':<30} {self.n_treated_units:>10}",
            f"{'Control units:':<30} {self.n_control_units:>10}",
            f"{'Treatment cohorts:':<30} {len(self.groups):>10}",
            f"{'Time periods:':<30} {len(self.time_periods):>10}",
            "",
        ]

        # Survey design info
        if self.survey_metadata is not None:
            sm = self.survey_metadata
            lines.extend(_format_survey_block(sm, 85))

        # Variance-estimator label (Phase 1b vcov_type), with two suppression
        # gates mirroring DiDResults.summary() (results.py:213-226) and
        # ImputationDiDResults.summary():
        #   1. Bootstrap fits: fit() overwrites the reported SE/CI/p-value with
        #      bootstrap_results, so the analytical variance-family label would
        #      mislabel the inference source — surface "Inference method:
        #      bootstrap" + the replication count instead.
        #   2. Survey fits: _format_survey_block above already reports the design
        #      (weight type, strata/PSU counts, replicate method), so a parallel
        #      variance line would be redundant/misleading.
        # Default cluster=None still clusters at the unit column (Gardner GMM
        # sandwich, two_stage.py:1547), so cluster_name carries the unit column
        # and _format_vcov_label renders the unit-cluster CR1 label, not HC1.
        if self.bootstrap_results is not None:
            lines.append(f"{'Inference method:':<30} {'bootstrap':>15}")
            lines.append(
                f"{'Bootstrap replications:':<30} {self.bootstrap_results.n_bootstrap:>15}"
            )
            lines.append("")
        elif self.survey_metadata is None:
            from diff_diff.results import _format_vcov_label

            vcov_label = _format_vcov_label(
                self.vcov_type,
                cluster_name=self.cluster_name,
                n_clusters=self.n_clusters,
                n_obs=self.n_obs,
            )
            if vcov_label:
                lines.append(f"{'Variance estimator:':<30} {vcov_label:>15}")
                lines.append("")

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

    def to_dataframe(self, level: str = "event_study") -> pd.DataFrame:
        """
        Convert results to DataFrame.

        Parameters
        ----------
        level : str, default="event_study"
            Level of aggregation:
            - "event_study": Event study effects by relative time
            - "group": Group (cohort) effects
            - "observation": Per-observation treatment effects

        Returns
        -------
        pd.DataFrame
            Results as DataFrame.
        """
        if level == "observation":
            return self.treatment_effects.copy()

        elif level == "event_study":
            if self.event_study_effects is None:
                raise ValueError(
                    "Event study effects not computed. Aggregate post-fit "
                    "instead - results.aggregate('event_study') returns the "
                    "EventStudyResults container (on a bootstrapped fit, "
                    "re-fit with n_bootstrap=0 or use the deprecated "
                    "fit-time aggregate=; a result unpickled from a pre-3.9 "
                    "release carries no kit and must be re-fit)."
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
            if self.group_effects is None:
                raise ValueError(
                    "Group effects not computed. Aggregate post-fit instead "
                    "- results.aggregate('group') returns the "
                    "AggregationResult container (on a bootstrapped fit, "
                    "re-fit with n_bootstrap=0 or use the deprecated "
                    "fit-time aggregate=; a result unpickled from a pre-3.9 "
                    "release carries no kit and must be re-fit)."
                )
            rows = []
            for g, data in sorted(self.group_effects.items()):
                rows.append(
                    {
                        "group": g,
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

        else:
            raise ValueError(
                f"Unknown level: {level}. Use 'event_study', 'group', or 'observation'."
            )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert headline results to a dictionary.

        Provides flat aliases (``att``/``se``/``t_stat``/``p_value``/
        ``conf_int_lower``/``conf_int_upper``) plus variance-estimator metadata
        (``vcov_type``, optional ``cluster_name``/``n_clusters``, optional
        ``n_bootstrap``, ``inference_method``). Per-cohort / per-horizon detail is
        exposed via :meth:`to_dataframe`. ``inference_method`` reports
        ``"cluster"`` for the default fit because the Gardner GMM sandwich
        clusters at the unit column (``n_clusters`` populated) — consistent with
        the CR1-at-unit summary label.

        Returns
        -------
        Dict[str, Any]
            Headline overall ATT plus inference metadata.
        """
        result: Dict[str, Any] = {
            "att": self.overall_att,
            "se": self.overall_se,
            "t_stat": self.overall_t_stat,
            "p_value": self.overall_p_value,
            "conf_int_lower": self.overall_conf_int[0],
            "conf_int_upper": self.overall_conf_int[1],
            "n_obs": self.n_obs,
            "n_treated_obs": self.n_treated_obs,
            "n_untreated_obs": self.n_untreated_obs,
            "n_treated_units": self.n_treated_units,
            "n_control_units": self.n_control_units,
            "alpha": self.alpha,
            "anticipation": self.anticipation,
            "vcov_type": self.vcov_type,
        }
        if self.cluster_name is not None:
            result["cluster_name"] = self.cluster_name
        if self.n_clusters is not None:
            result["n_clusters"] = self.n_clusters
        if self.bootstrap_results is not None:
            result["n_bootstrap"] = self.bootstrap_results.n_bootstrap
            result["inference_method"] = "bootstrap"
        elif self.survey_metadata is not None:
            result["inference_method"] = "survey"
        elif self.n_clusters is not None:
            result["inference_method"] = "cluster"
        else:
            result["inference_method"] = "analytical"
        return result

    @property
    def is_significant(self) -> bool:
        """Check if overall ATT is significant."""
        return bool(self.overall_p_value < self.alpha)

    @property
    def significance_stars(self) -> str:
        """Significance stars for overall ATT."""
        return _get_significance_stars(self.overall_p_value)
