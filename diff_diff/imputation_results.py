"""
Result containers for the Imputation DiD estimator.

This module contains ImputationBootstrapResults and ImputationDiDResults
dataclasses. Extracted from imputation.py for module size management.
"""

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from diff_diff.aggregation import AggregationMixin, AggregationResult, build_total_relay_row
from diff_diff.imputation_aggregation import _ImputationAggregationMixin
from diff_diff.results import _format_survey_block, _get_significance_stars
from diff_diff.results_base import BaseResults, _require_fit_alpha, build_event_study_surface


class _ImputationKitAggregator(_ImputationAggregationMixin):
    """Throwaway per-call host for the post-fit recompute (M-021/M-118).

    Hosts the moved aggregation methods with exactly the mixin's declared
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
        anticipation: int,
        horizon_max: Optional[int],
        pretrends: bool,
        aux_partition: str,
        leave_one_out: bool,
        rank_deficient_action: str,
        df_convention: str,
    ) -> None:
        self.alpha = alpha
        self.anticipation = anticipation
        self.horizon_max = horizon_max
        self.pretrends = pretrends
        self.aux_partition = aux_partition
        self.leave_one_out = leave_one_out
        self.rank_deficient_action = rank_deficient_action
        self.df_convention = df_convention


__all__ = [
    "ImputationBootstrapResults",
    "ImputationDiDResults",
]


@dataclass
class ImputationBootstrapResults:
    """
    Results from ImputationDiD bootstrap inference.

    Bootstrap is a library extension beyond Borusyak et al. (2024), which
    proposes only analytical inference via the conservative variance estimator.
    Provided for consistency with CallawaySantAnna and SunAbraham.

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
class ImputationDiDResults(BaseResults, AggregationMixin):
    """
    Results from Borusyak-Jaravel-Spiess (2024) imputation DiD estimation.

    Attributes
    ----------
    treatment_effects : pd.DataFrame
        Unit-level treatment effects with columns: unit, time, tau_hat, weight.
    overall_att : float
        Overall average treatment effect on the treated.
    overall_se : float
        Standard error of overall ATT.
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
        Number of treated observations (:math:`|\\Omega_1|`).
    n_untreated_obs : int
        Number of untreated observations (:math:`|\\Omega_0|`).
    n_treated_units : int
        Number of ever-treated units.
    n_control_units : int
        Number of units contributing to Omega_0.
    alpha : float
        Significance level used.
    pretrend_results : dict, optional
        Populated by pretrend_test().
    bootstrap_results : ImputationBootstrapResults, optional
        Bootstrap inference results.
    event_study_df : float or None
        Scalar survey df governing the event-study rows' t-inference (the
        FINAL replicate-override value on replicate fits; leads included
        on survey fits). ``None`` on non-survey fits, on bootstrapped
        fits, when no fit-time event-study surface was built, and for the
        replicate-undefined ``0`` sentinel.
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
    pretrend_results: Optional[Dict[str, Any]] = field(default=None, repr=False)
    bootstrap_results: Optional[ImputationBootstrapResults] = field(default=None, repr=False)
    # Internal: stores data needed for pretrend_test()
    _estimator_ref: Optional[Any] = field(default=None, repr=False)
    # Survey design metadata (SurveyMetadata instance from diff_diff.survey)
    survey_metadata: Optional[Any] = field(default=None, repr=False)
    # Variance-estimator metadata (Phase 1b interstitial #3).
    # vcov_type is permanently narrow to {"hc1"} per the IF-based variance
    # contract (see REGISTRY.md). cluster_name + n_clusters are populated
    # only under bare cluster=; suppressed under survey designs (the survey
    # block in summary() already renders the design's PSU/strata metadata).
    vcov_type: str = field(default="hc1")
    cluster_name: Optional[str] = field(default=None)
    n_clusters: Optional[int] = field(default=None)
    # BJS 2024 Supp. App. A.9 leave-one-out finite-sample variance refinement
    # (opt-in). Recorded here so reported SEs are self-describing.
    leave_one_out: bool = field(default=False)
    # The estimator's df_convention configuration echoed onto the results
    # ("residual" | "cluster" | "normal"; added 3.9). It governs only the
    # pretrends lead regression's per-lead t/p/CI - the BJS overall /
    # post-treatment inference is knob-independent.
    df_convention: Optional[str] = None
    # Private panel-backed post-fit aggregation kit (rows M-021/M-118),
    # attached by ImputationDiD.fit(). None on results unpickled from a
    # pre-3.9 release (aggregate() then fails with the re-fit message).
    # New fields are appended AFTER this one (the generated __init__
    # positional indexes are public API).
    _aggregation_kit: Optional[Any] = field(default=None, repr=False, compare=False)
    # Scalar survey df governing the event-study rows' t-inference (leads
    # included on survey fits). None on non-survey fits, on bootstrapped
    # fits (percentile inference used no df), when no event-study surface
    # was built, and for the replicate-undefined 0 sentinel. Appended last
    # per the positional-__init__ convention above.
    event_study_df: Optional[float] = None

    # Post-fit aggregation vocabulary (M-021). balance_e keeps the mixin
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

    # --- Post-fit aggregation (M-021/M-118) ------------------------------

    def _aggregate_compute(
        self, level: str, *, weights: Optional[str], balance_e: Optional[int]
    ) -> Any:
        kit = self._aggregation_kit
        if kit is None:
            raise ValueError(
                "This ImputationDiDResults carries no aggregation kit - it is "
                "attached by ImputationDiD.fit(), so a result unpickled from "
                "an older release will not have one. Re-fit with "
                "diff-diff >= 3.9 to aggregate post-fit."
            )
        # Per-level bootstrap policy (v4-design section 6, converged with row
        # M-027): 'simple' is a bit-exact RELAY of the stored overall quintet
        # - faithful under any inference regime, bootstrap included - so it
        # dispatches BEFORE the bootstrap gate. Only the RECOMPUTE levels
        # below fail closed on bootstrapped fits. (This supersedes the
        # uniform-conservatism decision recorded with M-021; its rationale is
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
        if level == "event_study" and bk["uses_replicate"] and bk["pretrends"]:
            # The same unsupported combination fit(aggregate='event_study')
            # rejects: the pre-period lead regression's per-replicate refits
            # are not implemented, and the replicate replay re-estimates
            # post-treatment targets only.
            raise NotImplementedError(
                "aggregate('event_study') is not available on this fit: it "
                "used pretrends=True with a replicate-weight survey design, "
                "and the pre-period lead regression's per-replicate refits "
                "are not yet implemented (fit(aggregate='event_study') "
                "rejects the same combination). Re-fit with pretrends=False, "
                "or use an analytical (strata/PSU/FPC) survey design."
            )
        # Fresh throwaway host per call, populated from KIT snapshots only
        # (estimator/config mutation after fit() must not leak in), and a
        # call-local projection cache (the id()-keyed cache keys are the
        # kit's own mask objects, so reuse within the call is exact).
        agg = _ImputationKitAggregator(
            alpha=kit.alpha,
            anticipation=kit.anticipation,
            horizon_max=bk["horizon_max"],
            pretrends=bk["pretrends"],
            aux_partition=bk["aux_partition"],
            leave_one_out=bk["leave_one_out"],
            rank_deficient_action=bk["rank_deficient_action"],
            df_convention=bk["df_convention"],
        )
        proj_cache: Dict[Any, Any] = {}
        common: Dict[str, Any] = dict(
            df=bk["df"],
            outcome=bk["outcome"],
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
            survey_df=bk["survey_df_seed"],
            resolved_survey=(None if bk["uses_replicate"] else bk["resolved_survey"]),
            proj_cache=proj_cache,
        )
        if level == "group":
            effects = agg._aggregate_group(**common)
            if bk["uses_replicate"]:
                # LEVEL-MATCHED replay: [overall, groups] - reproduces
                # fit(aggregate='group') exactly (see the replay docstring).
                agg._replicate_override_aggregates(
                    df=bk["df"],
                    outcome=bk["outcome"],
                    unit=bk["unit"],
                    time=bk["time"],
                    first_treat=bk["first_treat"],
                    covariates=bk["covariates"],
                    omega_0_mask=bk["omega_0_mask"],
                    omega_1_mask=bk["omega_1_mask"],
                    resolved_survey=bk["resolved_survey"],
                    overall_att=bk["overall_att"],
                    event_study_effects=None,
                    group_effects=effects,
                    balance_e=None,
                    survey_df_seed=bk["survey_df_seed"],
                )
            return self._group_effects_to_aggregation(effects, kit)
        # level == "event_study" (the mixin validated the vocabulary)
        es = agg._aggregate_event_study(**common, balance_e=balance_e)
        replay_df: Optional[int] = None
        if bk["uses_replicate"]:
            _, _, replay_df = agg._replicate_override_aggregates(
                df=bk["df"],
                outcome=bk["outcome"],
                unit=bk["unit"],
                time=bk["time"],
                first_treat=bk["first_treat"],
                covariates=bk["covariates"],
                omega_0_mask=bk["omega_0_mask"],
                omega_1_mask=bk["omega_1_mask"],
                resolved_survey=bk["resolved_survey"],
                overall_att=bk["overall_att"],
                event_study_effects=es,
                group_effects=None,
                balance_e=balance_e,
                survey_df_seed=bk["survey_df_seed"],
            )
        # Carrier + shared builder: ImputationDiD is a _from_relative_dict
        # producer, so the recomputed dict rides the same route as the
        # fit-time surface (zero-count-sentinel reference marking,
        # n_kind="obs"). The carrier's metadata is a copy-on-use of the
        # KIT's fit-final metadata copy (never the mutable public field);
        # on a replicate replay its df_survey is the REPLAYED level-matched
        # value, normalized by the same rule fit applies.
        meta = bk["survey_metadata"]
        if meta is not None:
            if bk["uses_replicate"]:
                meta = dataclasses.replace(
                    meta, df_survey=(replay_df if replay_df and replay_df > 0 else None)
                )
            else:
                meta = dataclasses.replace(meta)
        # Per-row df provenance: the df this route's ES rows actually used —
        # the level-matched replay value on replicate replays (the fit-time
        # snapshot came from the [overall]-only stack and can diverge), the
        # seed df otherwise; 0-sentinel normalized.
        _es_df = replay_df if bk["uses_replicate"] else bk["survey_df_seed"]
        carrier = dataclasses.replace(
            self,
            event_study_effects=es,
            survey_metadata=meta,
            anticipation=kit.anticipation,
            alpha=kit.alpha,
            event_study_df=(float(_es_df) if _es_df is not None and _es_df > 0 else None),
        )
        return build_event_study_surface(carrier)

    def _aggregate_simple_result(self, kit: Any) -> AggregationResult:
        """One-row relay of the stored overall inference (bit-exact).

        ``n = n_treated_obs`` (|Omega_1|) with ``n_kind="obs"``:
        ImputationDiD's ``n_treated_units``/``n_control_units`` unit sets
        OVERLAP (a treated unit with pre-periods counts in both), so the
        CS/EDiD disjoint-units convention cannot apply (the StackedDiD
        carve-out class); the treated-observation count is the population
        the overall ATT averages over and matches every other Imputation
        row's n semantics. Of that population, only finite-tau-hat
        observations enter the average - ``n`` reports the raw count, so
        on partially unidentified fits (which warn at fit time) ``n``
        exceeds the averaged support.

        ``df`` is the kit's ``survey_df_final`` snapshot - the exact value
        the STORED overall ``safe_inference`` received (on a replicate fit
        that value came from the ``[overall]``-only joint stack, which is
        precisely why it must be snapshotted rather than re-derived).
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
        mass, with ``C`` the FINITE-tau complete-case support snapshot
        (``total_support``, stashed at kit build) - the finite support is
        what the overall averages over, so ``C x overall = sum(tau)``
        exactly, fixing the documented raw-|Omega_1| overcount of the
        ``scale="auto"`` route ('simple''s ``n`` stays raw by contract).
        Zero support (the all-unidentified fit, where overall is already
        NaN) maps to a NaN mass -> all-NaN row. Fails closed on fits
        declaring a ``survey_design=`` (see the REGISTRY Note).
        """
        bk = kit.bookkeeping
        support = bk.get("total_support")
        if support is None:
            raise NotImplementedError(
                "aggregate('total') needs fit-time state this result "
                "predates: it was fitted before aggregate('total') existed - "
                "refit to use it."
            )
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
        mass = float(support) if support > 0 else float(np.nan)
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
        self, effects: Dict[Any, Dict[str, Any]], kit: Any
    ) -> AggregationResult:
        """Per-cohort AggregationResult from the recomputed group dict.

        ``df`` relays the PER-ROW ``df_used`` key each row's
        ``safe_inference`` recorded (capture-at-use: the analytical writer
        and the replicate override genuinely use different values on
        replicate fits; the all-NaN cohort branch writes no key, read via
        ``.get`` → NaN). ``weight=None``: cohort means over their own
        observations carry no cross-cohort mass (the CS rationale).
        ``n_kind="obs"``: ``n_obs`` counts the cohort's treated
        observations, matching the ES surface's n semantics.
        """
        labels = list(effects.keys())
        df_arr = np.array(
            [
                (np.nan if effects[g].get("df_used") is None else float(effects[g]["df_used"]))
                for g in labels
            ],
            dtype=float,
        )
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
            df=df_arr,
            alpha=kit.alpha,
            n_kind="obs",
            weight=None,
            estimator=type(self).__name__.replace("Results", ""),
        )

    def __repr__(self) -> str:
        """Concise string representation."""
        sig = _get_significance_stars(self.overall_p_value)
        return (
            f"ImputationDiDResults(ATT={self.overall_att:.4f}{sig}, "
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
            Accepted for signature uniformity. The stored intervals were
            computed at fit time; a value different from the stored
            ``alpha`` raises ValueError rather than silently recomputing
            or relabeling (bootstrap percentile intervals cannot be
            reconstructed from the reported SE). Re-fit at the desired
            alpha instead.

        Returns
        -------
        str
            Formatted summary.
        """
        alpha = _require_fit_alpha(alpha, self.alpha)
        conf_level = int((1 - alpha) * 100)

        lines = [
            "=" * 85,
            "Imputation DiD Estimator Results (Borusyak et al. 2024)".center(85),
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

        # Inference / variance metadata. Two suppression rules — match the
        # canonical DiDResults pattern at diff_diff/results.py:213-226:
        #   1. Survey designs: the survey block above already names the
        #      design + n_psu + df; the analytical SE is TSL on the combined
        #      IF (or replicate reweighting), not the raw HC1/CR1 sandwich.
        #   2. Bootstrap fits: fit() overwrites the reported SE/CI/p-value
        #      with bootstrap_results, so the analytical variance-family
        #      label would misstate the actual inference source. Surface an
        #      "Inference method: bootstrap" + replication count instead.
        if self.bootstrap_results is not None:
            lines.append(f"{'Inference method:':<30} {'bootstrap':>15}")
            lines.append(
                f"{'Bootstrap replications:':<30} {self.bootstrap_results.n_bootstrap:>15}"
            )
        elif self.survey_metadata is None:
            # Analytical, non-survey path: render the variance-family label.
            # For cluster=None ImputationDiD still clusters at unit by default
            # (Theorem 3 equation 7 conservative variance on per-unit IF
            # sums), so cluster_name is populated with the unit column name
            # and _format_vcov_label renders the unit-cluster CR1 label.
            from diff_diff.results import _format_vcov_label

            vcov_label = _format_vcov_label(
                self.vcov_type,
                cluster_name=self.cluster_name,
                n_clusters=self.n_clusters,
                n_obs=self.n_obs,
            )
            if vcov_label:
                lines.append(f"{'Variance estimator:':<30} {vcov_label:>15}")
        if self.n_clusters is not None and self.bootstrap_results is None:
            lines.append(f"{'Number of clusters:':<30} {self.n_clusters:>15}")
        if self.leave_one_out:
            lines.append(f"{'Leave-one-out variance:':<30} {'A.9 (BJS 2024)':>15}")

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

        # Pre-trend test
        if self.pretrend_results is not None:
            pt = self.pretrend_results
            lines.extend(
                [
                    "-" * 85,
                    "Pre-Trend Test (Equation 9)".center(85),
                    "-" * 85,
                    f"{'F-statistic:':<30} {pt['f_stat']:>10.3f}",
                    f"{'P-value:':<30} {pt['p_value']:>10.4f}",
                    f"{'Degrees of freedom:':<30} {pt['df']:>10}",
                    f"{'Number of leads:':<30} {pt['n_leads']:>10}",
                    "-" * 85,
                    "",
                ]
            )

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

    def to_dataframe(self, level: str = "observation") -> pd.DataFrame:
        """
        Convert results to DataFrame.

        Parameters
        ----------
        level : str, default="observation"
            Level of aggregation:
            - "observation": Unit-level treatment effects
            - "event_study": Event study effects by relative time
            - "group": Group (cohort) effects

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
                f"Unknown level: {level}. Use 'observation', 'event_study', or 'group'."
            )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert results to a dictionary.

        Provides flat headline aliases (``att``/``se``/``t_stat``/``p_value``/
        ``conf_int_lower``/``conf_int_upper``) plus variance-estimator
        metadata (``vcov_type``, optional ``cluster_name``/``n_clusters``,
        optional ``n_bootstrap``, ``inference_method``).

        Returns
        -------
        Dict[str, Any]
            Dictionary containing the headline overall ATT and inference
            metadata. Per-cohort / per-horizon detail is exposed via
            :meth:`to_dataframe`.
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
            "leave_one_out": self.leave_one_out,
        }
        if self.cluster_name is not None:
            result["cluster_name"] = self.cluster_name
        if self.n_clusters is not None:
            result["n_clusters"] = self.n_clusters
        if self.df_convention is not None:
            result["df_convention"] = self.df_convention
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

    def pretrend_test(self, n_leads: Optional[int] = None) -> Dict[str, Any]:
        """
        Run a pre-trend test (Equation 9 of Borusyak et al. 2024).

        Adds pre-treatment lead indicators to the Step 1 OLS and tests
        their joint significance via a Wald F-test (cluster-robust, or
        design-based survey VCV when survey_design was provided at fit).

        Parameters
        ----------
        n_leads : int, optional
            Number of pre-treatment leads to include. If None, uses all
            available pre-treatment periods minus one (for the reference period).

        Returns
        -------
        dict
            Dictionary with keys: 'f_stat', 'p_value', 'df', 'n_leads',
            'lead_coefficients'.
        """
        if self._estimator_ref is None:
            raise RuntimeError(
                "Pre-trend test requires internal estimator reference. "
                "Re-fit the model to use this method."
            )
        result = self._estimator_ref._pretrend_test(n_leads=n_leads)
        self.pretrend_results = result
        return result

    @property
    def is_significant(self) -> bool:
        """Check if overall ATT is significant."""
        return bool(self.overall_p_value < self.alpha)

    @property
    def significance_stars(self) -> str:
        """Significance stars for overall ATT."""
        return _get_significance_stars(self.overall_p_value)
