"""
Result containers for the de Chaisemartin-D'Haultfoeuille (dCDH) estimator.

This module contains ``ChaisemartinDHaultfoeuilleResults`` and
``DCDHBootstrapResults`` dataclasses produced by the
``ChaisemartinDHaultfoeuille`` (alias ``DCDH``) estimator. The dCDH
estimator is the most general library estimator for non-absorbing
(reversible) treatments (``LPDiD`` and ``TROP`` also support non-absorbing
treatment under stronger assumptions; see their ``non_absorbing`` parameters).
Phase 1 ships the contemporaneous-switch case ``DID_M`` (= ``DID_1`` of the
dynamic companion paper).

References
----------
- de Chaisemartin, C. & D'Haultfoeuille, X. (2020). Two-Way Fixed Effects
  Estimators with Heterogeneous Treatment Effects. *American Economic
  Review*, 110(9), 2964-2996.
- de Chaisemartin, C. & D'Haultfoeuille, X. (2022, revised July 2023).
  Difference-in-Differences Estimators of Intertemporal Treatment Effects.
  NBER Working Paper 29873.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from diff_diff.honest_did import HonestDiDResults

import numpy as np
import pandas as pd

from diff_diff.results import _get_significance_stars
from diff_diff.results_base import BaseResults

__all__ = [
    "ChaisemartinDHaultfoeuilleResults",
    "DCDHBootstrapResults",
]


@dataclass
class DCDHBootstrapResults:
    """
    Results from ChaisemartinDHaultfoeuille (dCDH) multiplier bootstrap inference.

    The bootstrap is a library extension beyond the dCDH papers, which
    propose only the analytical cohort-recentered plug-in variance from
    Web Appendix Section 3.7.3 of the dynamic companion paper. Provided
    for consistency with CallawaySantAnna / ImputationDiD / TwoStageDiD.

    Per-target SE / CI / p-value are populated for the three scalar
    dCDH estimands implemented in Phase 1: overall (``DID_M``), joiners
    (``DID_+``), and leavers (``DID_-``). When a target is not available
    in the underlying data (e.g., no leavers), the matching fields are
    ``None``.

    **Phase 1 per-period placebo (L_max=None) bootstrap is NOT computed.**
    The dynamic companion paper Section 3.7.3 derives the cohort-recentered
    analytical variance for ``DID_l`` only, not for the per-period
    ``DID_M^pl``. The ``placebo_se`` / ``placebo_ci`` / ``placebo_p_value``
    fields below remain ``None`` for Phase 1. Multi-horizon placebos
    (``L_max >= 1``) have valid SE via ``placebo_horizon_ses`` - this is
    a library extension applying the same IF/variance structure to the
    placebo estimand (see REGISTRY.md dynamic placebo SE Note).

    Attributes
    ----------
    n_bootstrap : int
        Number of bootstrap iterations.
    weight_type : str
        Type of bootstrap weights: ``"rademacher"``, ``"mammen"``, or
        ``"webb"``.
    alpha : float
        Significance level used for confidence intervals.
    overall_se : float
        Bootstrap standard error for ``DID_M``.
    overall_ci : tuple of float
        Bootstrap confidence interval for ``DID_M``.
    overall_p_value : float
        Bootstrap p-value for ``DID_M``.
    joiners_se : float, optional
        Bootstrap SE for joiners-only ``DID_+`` (``None`` if no joiners).
    joiners_ci : tuple of float, optional
        Bootstrap CI for joiners-only ``DID_+``.
    joiners_p_value : float, optional
        Bootstrap p-value for joiners-only ``DID_+``.
    leavers_se : float, optional
        Bootstrap SE for leavers-only ``DID_-`` (``None`` if no leavers).
    leavers_ci : tuple of float, optional
        Bootstrap CI for leavers-only ``DID_-``.
    leavers_p_value : float, optional
        Bootstrap p-value for leavers-only ``DID_-``.
    placebo_se : float, optional
        ``None`` for the Phase 1 single-period placebo (``L_max=None``).
        Multi-horizon placebo bootstrap SE is on
        ``placebo_horizon_ses``.
    placebo_ci : tuple of float, optional
        ``None`` for single-period placebo. See ``placebo_horizon_cis``.
    placebo_p_value : float, optional
        ``None`` for single-period placebo. See
        ``placebo_horizon_p_values``.
    bootstrap_distribution : np.ndarray, optional
        Full bootstrap distribution of the overall ``DID_M`` estimator
        (shape: ``(n_bootstrap,)``). Stored for advanced diagnostics;
        suppressed from ``__repr__``.
    """

    n_bootstrap: int
    weight_type: str
    alpha: float
    overall_se: float
    overall_ci: Tuple[float, float]
    overall_p_value: float
    joiners_se: Optional[float] = None
    joiners_ci: Optional[Tuple[float, float]] = None
    joiners_p_value: Optional[float] = None
    leavers_se: Optional[float] = None
    leavers_ci: Optional[Tuple[float, float]] = None
    leavers_p_value: Optional[float] = None
    placebo_se: Optional[float] = None
    placebo_ci: Optional[Tuple[float, float]] = None
    placebo_p_value: Optional[float] = None
    bootstrap_distribution: Optional[np.ndarray] = field(default=None, repr=False)

    # --- Phase 2: per-horizon bootstrap ---
    event_study_ses: Optional[Dict[int, float]] = field(default=None, repr=False)
    event_study_cis: Optional[Dict[int, Tuple[float, float]]] = field(default=None, repr=False)
    event_study_p_values: Optional[Dict[int, float]] = field(default=None, repr=False)
    placebo_horizon_ses: Optional[Dict[int, float]] = field(default=None, repr=False)
    placebo_horizon_cis: Optional[Dict[int, Tuple[float, float]]] = field(default=None, repr=False)
    placebo_horizon_p_values: Optional[Dict[int, float]] = field(default=None, repr=False)
    cband_crit_value: Optional[float] = None

    # --- Phase 3: per-path bootstrap (by_path) ---
    # Keyed by path tuple -> horizon -> scalar/pair. Populated only when
    # by_path + n_bootstrap > 0 is active; `None` otherwise. Percentile
    # CI + percentile p-value per library Round-10 convention; caller
    # (fit()) propagates these to path_effects[path]["horizons"][l]
    # directly and computes a SE-derived t-stat via `safe_inference`.
    path_ses: Optional[Dict[Tuple[int, ...], Dict[int, float]]] = field(default=None, repr=False)
    path_cis: Optional[Dict[Tuple[int, ...], Dict[int, Tuple[float, float]]]] = field(
        default=None, repr=False
    )
    path_p_values: Optional[Dict[Tuple[int, ...], Dict[int, float]]] = field(
        default=None, repr=False
    )

    # --- Phase 3: per-path placebo bootstrap (by_path + placebo) ---
    # Same shape and library convention as path_ses / path_cis /
    # path_p_values, but for backward placebo lags (l = 1..L_max). Keyed
    # by **positive** int internally; the propagation block in fit()
    # writes them to path_placebo_event_study[path][-l] (negative key)
    # to match the placebo_event_study convention. Populated only when
    # by_path + placebo + n_bootstrap > 0 is active; `None` otherwise.
    path_placebo_ses: Optional[Dict[Tuple[int, ...], Dict[int, float]]] = field(
        default=None, repr=False
    )
    path_placebo_cis: Optional[Dict[Tuple[int, ...], Dict[int, Tuple[float, float]]]] = field(
        default=None, repr=False
    )
    path_placebo_p_values: Optional[Dict[Tuple[int, ...], Dict[int, float]]] = field(
        default=None, repr=False
    )

    # --- Phase 3: per-path joint sup-t critical values (by_path + n_bootstrap > 0) ---
    # Per-path sup-t simultaneous-band critical value `c_p =
    # quantile(max_l |t_l|, 1-alpha)` from a fresh shared-weights
    # multiplier-bootstrap draw per path. Naming parity with the OVERALL
    # `cband_crit_value` scalar at line 131 (singular -> plural since one
    # crit per path). Gates: a path appears only when (>=2 valid horizons
    # with finite bootstrap SE > 0) AND (a strict majority — more than
    # 50% — of sup-t draws are finite); paths failing either gate are
    # absent from the dict. `None` when bootstrap didn't run; empty dict
    # when ran but no path passed both gates.
    path_cband_crit_values: Optional[Dict[Tuple[int, ...], float]] = field(default=None, repr=False)
    path_cband_n_valid_horizons: Optional[Dict[Tuple[int, ...], int]] = field(
        default=None, repr=False
    )


@dataclass
class ChaisemartinDHaultfoeuilleResults(BaseResults):
    """
    Results from de Chaisemartin-D'Haultfoeuille (dCDH) Phase 1 estimation.

    Phase 1 ships the contemporaneous-switch estimator ``DID_M`` (= ``DID_1``
    at horizon ``l = 1`` of the dynamic companion paper) plus the joiners-
    only / leavers-only views, the single-lag placebo ``DID_M^pl``, and
    optionally the TWFE decomposition diagnostic (per-cell weights,
    fraction negative, ``sigma_fe``).

    Notes
    -----
    The analytical confidence interval is **conservative** under
    Assumption 8 (independent groups) of the dynamic companion paper, and
    exact only under iid sampling. This is documented as a deliberate
    deviation from "default nominal coverage" in the methodology registry.

    For binary treatment in Phase 1, multi-switch groups (i.e., groups
    that switch treatment more than once) are dropped before estimation
    when ``drop_larger_lower=True`` (the default), matching the R
    ``DIDmultiplegtDYN`` reference. The number of dropped groups is
    exposed via ``n_groups_dropped_crossers``.

    **Inference-method switch when bootstrap is enabled.** The
    ``overall_p_value`` / ``overall_conf_int`` (and joiners/leavers
    analogues) fields are populated by *normal-theory* inference from
    the cohort-recentered analytical SE when ``n_bootstrap=0`` (the
    default). When ``n_bootstrap > 0``, the same fields are populated
    by *percentile-based bootstrap inference* from the multiplier
    bootstrap distribution computed by ``_compute_dcdh_bootstrap()``.
    The t-stat (``overall_t_stat``, etc.) is computed from the SE in
    both cases, since percentile bootstrap does not define an
    alternative t-stat semantic. ``event_study_effects[1]``,
    ``summary()``, ``to_dataframe()``, ``is_significant``, and
    ``significance_stars`` all read from these top-level fields and
    therefore reflect the bootstrap inference automatically. The
    single-period placebo (``L_max=None``) still has NaN bootstrap
    fields; multi-horizon placebos (``L_max >= 1``) have valid
    bootstrap SE/CI/p via ``placebo_horizon_ses/cis/p_values``.
    See the methodology registry
    ``Note (bootstrap inference surface)`` for the full contract and
    library precedent.

    Attributes
    ----------
    overall_att : float
        ``DID_M = DID_1``: the contemporaneous-switch dCDH point estimate.
    overall_se : float
        Standard error of ``DID_M``.
    overall_t_stat : float
    overall_p_value : float
    overall_conf_int : tuple of float
    joiners_att : float
        ``DID_+``: the joiners-only contribution. ``NaN`` when
        ``joiners_available`` is False.
    joiners_se : float
    joiners_t_stat : float
    joiners_p_value : float
    joiners_conf_int : tuple of float
    n_joiner_cells : int
        Total number of joiner switching ``(g, t)`` cells across all
        periods. Each cell counted once. Equals
        ``sum_t (#{g : D_{g,t-1}=0, D_{g,t}=1})``.
    n_joiner_obs : int
        Total raw observation count across joiner cells, summing
        ``n_gt`` over the same set of cells. For balanced
        one-observation-per-cell panels this equals ``n_joiner_cells``;
        for individual-level inputs with multiple observations per
        ``(g, t)`` it can be larger.
    joiners_available : bool
        ``True`` if at least one joiner switching cell exists.
    leavers_att : float
        ``DID_-``: the leavers-only contribution. ``NaN`` when
        ``leavers_available`` is False.
    leavers_se : float
    leavers_t_stat : float
    leavers_p_value : float
    leavers_conf_int : tuple of float
    n_leaver_cells : int
        Total number of leaver switching ``(g, t)`` cells (mirror of
        ``n_joiner_cells``).
    n_leaver_obs : int
        Total raw observation count across leaver cells (mirror of
        ``n_joiner_obs``).
    leavers_available : bool
    placebo_effect : float
        ``DID_M^pl``: the single-lag placebo. ``NaN`` when
        ``placebo_available`` is False.
    placebo_se : float
    placebo_t_stat : float
    placebo_p_value : float
    placebo_conf_int : tuple of float
    placebo_available : bool
        ``True`` when ``T >= 3`` and at least one qualifying placebo cell
        exists.
    per_period_effects : dict
        Per-period decomposition. Keys are period values; each value is a
        dict with the following keys:

        - ``"did_plus_t"`` (float): joiner effect at this period
          (``0.0`` if no joiners or A11 violation)
        - ``"did_minus_t"`` (float): leaver effect at this period
        - ``"n_10_t"`` (int): joiner cell count
        - ``"n_01_t"`` (int): leaver cell count
        - ``"n_00_t"`` (int): stable-untreated cell count
        - ``"n_11_t"`` (int): stable-treated cell count
        - ``"did_plus_t_a11_zeroed"`` (bool): True when joiners exist but
          no stable-untreated controls (Assumption 11 violation, period
          contributes 0 to numerator with non-zero weight in denominator)
        - ``"did_minus_t_a11_zeroed"`` (bool): mirror for leavers
    twfe_weights : pd.DataFrame, optional
        Per-cell TWFE decomposition weights from Theorem 1 of de
        Chaisemartin & D'Haultfoeuille (2020). Columns: ``group``,
        ``time``, ``weight``. Computed on the **FULL pre-filter cell
        sample** passed by the user (the same input the standalone
        :func:`twowayfeweights` function uses) — NOT the post-filter
        estimation sample described by ``overall_att`` and
        ``groups``. When ``fit()`` drops groups via the ragged-panel
        or ``drop_larger_lower`` filters, ``results.twfe_*`` and
        ``results.overall_att`` describe different samples and a
        ``UserWarning`` is emitted; see REGISTRY.md
        ``ChaisemartinDHaultfoeuille`` ``Note (TWFE diagnostic
        sample contract)`` for the rationale. Only populated when
        ``twfe_diagnostic=True``.
    twfe_fraction_negative : float, optional
        Fraction of treated-cell weights that are negative. ``> 0`` is
        the diagnostic for the heterogeneous-treatment-effect bias of
        the plain TWFE estimator on the **FULL pre-filter cell sample**
        (NOT the post-filter estimation sample). See the
        ``twfe_weights`` docstring above for the sample contract.
    twfe_sigma_fe : float, optional
        Smallest standard deviation of per-cell treatment effects that
        could flip the sign of the plain TWFE estimator (Corollary 1 of
        the AER 2020 paper). Computed on the **FULL pre-filter cell
        sample**.
    twfe_beta_fe : float, optional
        The plain TWFE coefficient computed on the **FULL pre-filter
        cell sample**, for comparison with ``overall_att``. Note that
        the two are computed on different samples when ``fit()``
        filters drop groups — see the ``twfe_weights`` docstring above
        for the sample contract.
    groups : list
        Group identifiers in the post-filter sample.
    time_periods : list
        Time periods in the panel.
    n_obs : int
        Total observations after filtering.
    n_treated_obs : int
        Treated observations in the post-filter sample.
    n_switcher_cells : int
        When ``L_max=None``: number of switching ``(g, t)`` cells
        (``N_S = sum_t (n_10_t + n_01_t)``). When ``L_max >= 1``:
        number of eligible switcher groups at horizon 1 (``N_1``).
        Previously this field always held the cell count; for
        ``L_max >= 1`` it was repurposed to hold the per-group count
        that matches the ``DID_1`` estimand. Originally equals
        once regardless of how many original observations fed into it.
        This is the ``N_S`` denominator of ``DID_M`` under the library's
        equal-cell weighting convention (cell counts, not within-cell
        observation sums). The AER 2020 paper's Equation 3 defines
        ``N_{d,d',t} = sum_g N_{g,t}`` (observation sums); the
        library's choice is a documented deviation - see
        ``docs/methodology/REGISTRY.md`` ``## ChaisemartinDHaultfoeuille``
        L517 for the full Note.
    n_cohorts : int
        Distinct cohorts ``(D_{g,1}, F_g, S_g)`` after filtering.
    n_groups_dropped_crossers : int
        Number of groups dropped because they were multi-switch (matches
        R's ``drop_larger_lower=TRUE`` behavior). ``0`` when
        ``drop_larger_lower=False`` or no crossers exist.
    n_groups_dropped_singleton_baseline : int
        Number of groups whose baseline ``D_{g,1}`` is unique in the
        post-drop panel (footnote 15 of the dynamic paper). They are
        excluded from the cohort-recentered VARIANCE computation only —
        they remain in the point-estimate sample as period-based stable
        controls (see REGISTRY.md ``ChaisemartinDHaultfoeuille`` for the
        period-vs-cohort deviation that makes this distinction matter).
    n_groups_dropped_never_switching : int
        Number of groups with ``S_g = 0`` (never switched). **Reported
        for backwards compatibility only.** Per the Round 2 full
        influence-function fix, never-switching groups are NOT excluded
        from the variance: they contribute via their stable-control
        roles in the per-period IF formula. The field name retains
        "dropped" for API stability but no actual exclusion happens.
    alpha : float
        Significance level used for confidence intervals.
    event_study_effects : dict, optional
        Populated with horizon ``1`` when ``L_max=None``, or horizons
        ``1..L_max`` when ``L_max >= 1``. When ``L_max >= 1``, uses the
        per-group ``DID_{g,l}`` path; when ``L_max=None``, uses the
        per-period ``DID_M`` path.
    event_study_df : float, optional
        Inference degrees-of-freedom PROVENANCE for the event-study and
        placebo surfaces: the single df every stored row's
        ``safe_inference`` received. One scalar suffices because both
        surfaces are computed from the same design df (and are refreshed
        together to the final effective df under replicate weights).
        ``None`` when the rows used normal theory (no survey design), when
        the df was undefined, or when bootstrap overrode the stored
        inference with percentile values that never used a df.
    normalized_effects : dict, optional
        Normalized estimator ``DID^n_l``. Populated when ``L_max >= 1``.
    cost_benefit_delta : dict, optional
        Cost-benefit aggregate ``delta``. Populated when ``L_max >= 2``.
    sup_t_bands : dict, optional
        Sup-t simultaneous confidence-band metadata for the OVERALL
        event-study surface. Holds ``{"crit_value": float, "alpha":
        float, "n_bootstrap": int, "method": str}``. Populated when
        ``n_bootstrap > 0`` AND there are at least 2 valid horizons
        with finite bootstrap SE > 0 AND a strict majority (more than
        50%) of sup-t draws are finite. The band itself is written
        per-horizon as
        ``cband_conf_int`` on ``event_study_effects[l]``. ``None``
        otherwise. Python-only library extension; R
        ``did_multiplegt_dyn`` provides no joint / sup-t bands.
    covariate_residuals : pd.DataFrame, optional
        ``DID^X`` first-stage diagnostics: per-baseline ``theta_hat``,
        ``n_obs``, and ``r_squared``. Populated when ``controls`` is set.
    linear_trends_effects : dict, optional
        Cumulated ``DID^{fd}`` level effects ``delta^{fd}_l``. Keyed by
        horizon. Populated when ``trends_linear=True``.
    heterogeneity_effects : dict, optional
        Per-horizon heterogeneity test results ``beta^{het}_l``.
        Populated when ``heterogeneity`` is set.
    design2_effects : dict, optional
        Design-2 switch-in/switch-out descriptive summary. Populated
        when ``design2=True``.
    path_effects : dict, optional
        Per-path event-study effects keyed by observed treatment
        trajectory (tuple of int). Populated when ``by_path`` is a
        positive int OR ``paths_of_interest`` is a list of int tuples
        at estimator construction. Each entry holds
        ``{"n_groups": int, "frequency_rank": int,
        "horizons": {l: {"effect", "se", "t_stat", "p_value",
        "conf_int", "n_obs"}}}`` for ``l = 1..L_max``. Under
        ``paths_of_interest``, dict-insertion order matches the user-
        specified path order; ``frequency_rank`` is the within-
        selected-paths rank by descending observed-group count
        (decoupled from iteration order).
    path_placebo_event_study : dict, optional
        Per-path backward-horizon placebos ``DID^{pl}_{path, l}`` for
        ``l = 1..L_max``, keyed by observed treatment trajectory (tuple
        of int). Inner dict keys are **negative** ints (``-l`` for lag
        ``l``) to mirror the ``placebo_event_study`` convention so a
        unified ``{**path_effects[p]["horizons"],
        **path_placebo_event_study[p]}`` view is well-formed across
        forward and backward horizons. Each inner entry holds
        ``{"effect", "se", "t_stat", "p_value", "conf_int", "n_obs"}``.
        Populated when (``by_path`` is a positive int OR
        ``paths_of_interest`` is set) AND ``placebo=True`` AND
        ``L_max >= 1``. Empty-state contract mirrors ``path_effects``:
        ``None`` when ``by_path / paths_of_interest + placebo`` was
        not requested; ``{}`` when requested but no observed path has
        a complete window ``[F_g-1, F_g-1+L_max]`` within the
        panel (the same regime where ``path_effects`` returns ``{}``,
        with the same ``UserWarning`` at fit-time). Downstream callers
        should distinguish the two states. Inherits the cross-path
        cohort-sharing SE deviation from R documented for
        ``path_effects``. See REGISTRY.md
        ``Note (Phase 3 by_path ...)`` → "Per-path placebos".
    path_heterogeneity_effects : dict, optional
        Per-path heterogeneity test results (Web Appendix Section 1.5,
        Lemma 7) when ``heterogeneity`` is set AND (``by_path=k`` or
        ``paths_of_interest=[(...), ...]``) is set. Inner dict keyed by
        horizon directly (no ``"horizons"`` wrapper); each entry holds
        ``{"beta", "se", "t_stat", "p_value", "conf_int", "n_obs"}``,
        where ``beta`` is the heterogeneity coefficient on the path-
        restricted switcher subsample - plain OLS on the non-survey
        path, WLS-on-pweights under ``survey_design``. Cohort
        dummies in the design matrix absorb baseline by construction.
        Empty-state contract mirrors ``path_effects``: ``None`` when not
        requested; ``{}`` when requested but no path has eligible
        switchers. Mirrors R ``did_multiplegt_dyn(..., by_path,
        predict_het)`` per-by_level dispatch. See REGISTRY.md
        ``Note (Phase 3 by_path ...)`` → "Per-path heterogeneity testing".
    path_cumulated_event_study : dict, optional
        Per-path cumulated level effects ``delta_{path, l} =
        sum_{l'=1..l} DID^{fd}_{path, l'}`` for ``l = 1..L_max``,
        keyed by observed treatment trajectory (tuple of int). Inner
        dict is keyed by horizon directly (no ``"horizons"`` wrapper);
        each entry holds ``{"effect", "se", "t_stat", "p_value",
        "conf_int", "n_obs"}``. Populated when (``by_path`` is a
        positive int OR ``paths_of_interest`` is set) AND
        ``trends_linear=True`` AND ``L_max >= 1``; ``None`` otherwise. Mirrors the global ``linear_trends_effects``
        cumulation: SE on the cumulated layer is the conservative
        upper bound (sum of per-horizon component SEs from
        ``path_effects[path]["horizons"][l]["se"]``, NaN-consistent).
        Built AFTER bootstrap propagation so the cumulated SE / t / p
        / CI are derived from the FINAL post-bootstrap per-horizon
        SEs when ``n_bootstrap > 0``. Surfaced as ``cumulated_effect``
        / ``cumulated_se`` columns on
        ``to_dataframe(level="by_path")`` (always-present, NaN-when-
        None) and as a per-path "Cumulated Level Effects" sub-section
        in ``summary()``. See REGISTRY.md ``Note (Phase 3 by_path
        ...)`` → "Per-path linear-trends DID^{fd}".
    path_sup_t_bands : dict, optional
        Per-path joint sup-t simultaneous-band metadata, keyed by
        observed treatment trajectory (tuple of int). Each entry holds
        ``{"crit_value": float, "alpha": float, "n_bootstrap": int,
        "method": str, "n_valid_horizons": int}``. Populated when
        (``by_path`` is a positive int OR ``paths_of_interest`` is
        set) AND ``n_bootstrap > 0``. The
        band itself is applied per-horizon as ``cband_conf_int`` on
        ``path_effects[path]["horizons"][l]`` and rendered as
        ``cband_lower`` / ``cband_upper`` columns on
        ``to_dataframe(level="by_path")``. Empty-state contract:
        ``None`` when not requested (no bootstrap, or both ``by_path``
        and ``paths_of_interest`` are ``None``); ``{}`` when requested
        but no path passed both gates (``>=2``
        valid horizons with finite bootstrap SE ``> 0`` AND a strict
        majority — more than 50% — of finite sup-t draws). Bands
        cover joint inference WITHIN a
        single path across horizons; they do NOT provide simultaneous
        coverage across paths. Inherits the cross-path cohort-sharing
        SE deviation from R documented for ``path_effects`` (the
        bootstrap SE used as the t-stat denominator carries the same
        deviation). Python-only library extension; R
        ``did_multiplegt_dyn`` provides no joint / sup-t bands at any
        surface. See REGISTRY.md ``Note (Phase 3 by_path per-path
        joint sup-t bands)``.
    honest_did_results : HonestDiDResults, optional
        HonestDiD sensitivity analysis bounds (Rambachan & Roth 2023).
        Populated when ``honest_did=True`` in ``fit()`` or by calling
        ``compute_honest_did(results)`` post-hoc. Contains identified
        set bounds, robust confidence intervals, and breakdown analysis.
    survey_metadata : Any, optional
        Populated when ``fit(..., survey_design=sd)`` is called; ``None``
        otherwise. Carries the resolved survey design summary
        (``weight_type``, strata/PSU counts, ``df_survey``, weight range,
        and replicate-method info when applicable). ``df_survey`` is
        threaded into survey-aware inference (t-distribution at all
        analytical surfaces) and consumed by ``compute_honest_did()`` to
        produce survey-aware critical values.
    bootstrap_results : DCDHBootstrapResults, optional
        Bootstrap inference results when ``n_bootstrap > 0``.
    """

    # --- Core: DID_M aggregate ---
    overall_att: float
    overall_se: float
    overall_t_stat: float
    overall_p_value: float
    overall_conf_int: Tuple[float, float]

    # --- Joiners-only view (DID_+) ---
    joiners_att: float
    joiners_se: float
    joiners_t_stat: float
    joiners_p_value: float
    joiners_conf_int: Tuple[float, float]
    n_joiner_cells: int
    n_joiner_obs: int
    joiners_available: bool

    # --- Leavers-only view (DID_-) ---
    leavers_att: float
    leavers_se: float
    leavers_t_stat: float
    leavers_p_value: float
    leavers_conf_int: Tuple[float, float]
    n_leaver_cells: int
    n_leaver_obs: int
    leavers_available: bool

    # --- Placebo (DID_M^pl) ---
    placebo_effect: float
    placebo_se: float
    placebo_t_stat: float
    placebo_p_value: float
    placebo_conf_int: Tuple[float, float]
    placebo_available: bool

    # --- Per-period decomposition ---
    per_period_effects: Dict[Any, Dict[str, Any]]

    # --- Metadata ---
    groups: List[Any]
    time_periods: List[Any]
    n_obs: int
    n_treated_obs: int
    n_switcher_cells: int
    n_cohorts: int
    n_groups_dropped_crossers: int
    n_groups_dropped_singleton_baseline: int
    n_groups_dropped_never_switching: int

    # --- Event study (Phase 2: multi-horizon) ---
    # Populated with {l: {effect, se, t_stat, p_value, conf_int, n_obs}}.
    # Phase 1 (L_max=None): single entry {1: {...}} mirroring overall_att.
    # Phase 2 (L_max>=2): entries for l = 1, ..., L_max.
    event_study_effects: Optional[Dict[int, Dict[str, Any]]] = None
    L_max: Optional[int] = None
    # Dynamic placebos DID^{pl}_l with negative horizon keys.
    # None in Phase 1; populated as {-1: {...}, -2: {...}} in Phase 2.
    placebo_event_study: Optional[Dict[int, Dict[str, Any]]] = field(default=None, repr=False)

    # --- TWFE decomposition diagnostic (Theorem 1 of AER 2020) ---
    twfe_weights: Optional[pd.DataFrame] = field(default=None, repr=False)
    twfe_fraction_negative: Optional[float] = None
    twfe_sigma_fe: Optional[float] = None
    twfe_beta_fe: Optional[float] = None

    alpha: float = 0.05

    # --- Forward-compat placeholders (always None in Phase 1) ---
    normalized_effects: Optional[Dict[int, Dict[str, Any]]] = field(default=None, repr=False)
    cost_benefit_delta: Optional[Dict[str, Any]] = field(default=None, repr=False)
    sup_t_bands: Optional[Dict[str, Any]] = field(default=None, repr=False)
    covariate_residuals: Optional[pd.DataFrame] = field(default=None, repr=False)
    linear_trends_effects: Optional[Dict[int, Dict[str, Any]]] = field(default=None, repr=False)
    # PR #347 R9 P1: persist the fit-time ``trends_linear`` flag
    # explicitly. The previous approach of inferring the flag from
    # ``linear_trends_effects is not None`` broke on the empty-
    # horizon case — the estimator sets ``linear_trends_effects=None``
    # when the cumulated dict is empty but still unconditionally
    # NaN-s ``overall_att`` for ``trends_linear=True`` with
    # ``L_max >= 2``. BR/DR dispatch on this flag (via
    # ``describe_target_parameter``) to route the no-scalar-by-design
    # headline correctly even when the horizon surface is empty.
    trends_linear: Optional[bool] = None
    heterogeneity_effects: Optional[Dict[int, Dict[str, Any]]] = field(default=None, repr=False)
    design2_effects: Optional[Dict[str, Any]] = field(default=None, repr=False)
    path_effects: Optional[Dict[Tuple[int, ...], Dict[str, Any]]] = field(default=None, repr=False)
    # Per-path backward-horizon placebos. Inner dict keys are NEGATIVE
    # ints (-l for lag l) to match `placebo_event_study`'s convention,
    # so a unified `{**path_effects[p]["horizons"],
    # **path_placebo_event_study[p]}` view is well-formed across both
    # forward and backward horizons within a single path.
    path_placebo_event_study: Optional[Dict[Tuple[int, ...], Dict[int, Dict[str, Any]]]] = field(
        default=None, repr=False
    )
    # Per-path heterogeneity test (Web Appendix Section 1.5, Lemma 7)
    # under `by_path` / `paths_of_interest`. Inner dict keyed by horizon
    # directly: `{path: {l: {beta, se, t_stat, p_value, conf_int, n_obs}}}`.
    # Mirrors the simpler `path_placebo_event_study` shape — no metadata
    # wrapper because frequency_rank / n_groups already live on
    # `path_effects[path]` for the same path. Empty-state contract
    # mirrors `path_effects`: None when not requested (no `heterogeneity`
    # kwarg or no `by_path` / `paths_of_interest` selector); `{}` when
    # requested but no path is observed.
    path_heterogeneity_effects: Optional[Dict[Tuple[int, ...], Dict[int, Dict[str, Any]]]] = field(
        default=None, repr=False
    )
    # Per-path cumulated event study (level effects under `trends_linear`
    # = True). `path_effects[path]["horizons"][l]` surfaces raw
    # `DID^{fd}_l` per path; this field surfaces the cumulated level
    # effect `delta_l = sum_{l'=1..l} DID^{fd}_{path, l'}` per (path,
    # horizon), mirroring the global `linear_trends_effects` cumulation
    # for non-by_path fits. Inner dict keyed by horizon directly (no
    # `horizons` wrapper). `None` when not requested (`by_path is None`
    # or `trends_linear=False`); `{}` is impossible (the field follows
    # `path_effects` so a populated `path_effects` plus `trends_linear`
    # always populates this). SE on the cumulated layer is the
    # conservative upper bound (sum of per-horizon component SEs,
    # NaN-consistent), matching the global `linear_trends_effects`
    # convention.
    path_cumulated_event_study: Optional[Dict[Tuple[int, ...], Dict[int, Dict[str, Any]]]] = field(
        default=None, repr=False
    )
    # Per-path joint sup-t simultaneous-band metadata. Keyed by path
    # tuple; each entry holds `{"crit_value", "alpha", "n_bootstrap",
    # "method", "n_valid_horizons"}`. Populated when EITHER `by_path` is
    # a positive int OR `paths_of_interest` is non-empty AND
    # `n_bootstrap > 0`. The joint band itself is written per-horizon as
    # `cband_conf_int` on `path_effects[path]["horizons"][l]` (mirrors
    # the OVERALL `event_study_effects[l]["cband_conf_int"]` pattern
    # populated alongside the bootstrap propagation in
    # `chaisemartin_dhaultfoeuille.py::fit`). Empty-state contract:
    # `None` when not requested (no bootstrap, or both `by_path` and
    # `paths_of_interest` are `None`); `{}` when requested but no path
    # passed both gates (>=2 valid horizons AND a strict majority — more
    # than 50% — of finite sup-t draws). The bands cover joint inference
    # WITHIN a single path across horizons; they do NOT provide
    # simultaneous coverage across paths.
    path_sup_t_bands: Optional[Dict[Tuple[int, ...], Dict[str, Any]]] = field(
        default=None, repr=False
    )
    honest_did_results: Optional["HonestDiDResults"] = field(default=None, repr=False)

    # --- Repr-suppressed metadata ---
    survey_metadata: Optional[Any] = field(default=None, repr=False)
    bootstrap_results: Optional[DCDHBootstrapResults] = field(default=None, repr=False)
    _estimator_ref: Optional[Any] = field(default=None, repr=False)

    # event_study_df (spec section 5, row M-092): the ONE inference df every
    # stored event-study AND placebo row's ``safe_inference`` actually
    # received - a single scalar because both surfaces are computed from the
    # same design df (and, under replicate weights, are re-run together to
    # the final effective df). None means the rows used normal theory (no
    # survey design), the df was undefined (the df<=0 replicate sentinel,
    # which yields NaN inference), or bootstrap overrode the stored p/CIs
    # with percentile values that never used a df. Note the bootstrap clear
    # drops the whole channel even when a partial override leaves some rows
    # analytic - a conservative under-claim, consistent with the other
    # producers.
    # Declared LAST so every pre-existing field keeps its positional index
    # in the generated __init__ (the constructor signature is public API).
    event_study_df: Optional[float] = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Repr / properties
    # ------------------------------------------------------------------

    def _has_trends_linear(self) -> bool:
        """Return whether ``trends_linear=True`` was active on this fit.

        PR #347 R9/R10: prefer the persisted fit-time flag. Fall back
        to ``linear_trends_effects is not None`` for legacy result
        objects that predate the persisted field. The inference
        fallback is correct on populated-surface fits but fails on
        empty-surface fits (``trends_linear=True``, ``L_max>=2`` with
        no estimable horizons) because ``linear_trends_effects`` is
        set to ``None`` in that case; the persisted flag handles it.
        """
        persisted = self.trends_linear
        if isinstance(persisted, bool):
            return persisted
        return self.linear_trends_effects is not None

    def _horizon_label(self, h) -> str:
        """Return per-horizon estimand label for event study rows."""
        has_controls = self.covariate_residuals is not None
        has_trends = self._has_trends_linear()
        if has_controls and has_trends:
            return f"DID^{{X,fd}}_{h}"
        elif has_controls:
            return f"DID^X_{h}"
        elif has_trends:
            return f"DID^{{fd}}_{h}"
        return f"DID_{h}"

    def _estimand_label(self) -> str:
        """Return the estimand label based on active features."""
        has_controls = self.covariate_residuals is not None
        has_trends = self._has_trends_linear()

        # When trends_linear + L_max>=2, overall is NaN (no aggregate).
        # Label reflects that per-horizon effects are in
        # linear_trends_effects — UNLESS that surface is also empty
        # (empty-horizon subcase; PR #347 R13 P1). In the
        # empty-surface subcase we do not direct users to a
        # nonexistent dict; we name the empty state instead.
        if has_trends and self.L_max is not None and self.L_max >= 2:
            base_label = "DID^{X,fd}_l" if has_controls else "DID^{fd}_l"
            if self.linear_trends_effects is None:
                return f"{base_label} (no cumulated level effects survived estimation)"
            return f"{base_label} (see linear_trends_effects)"

        if self.L_max is not None and self.L_max >= 2:
            base = "delta"
        elif self.L_max is not None and self.L_max == 1:
            base = "DID_1"
        else:
            base = "DID_M"

        if has_controls and has_trends:
            suffix = "^{X,fd}"
        elif has_controls:
            suffix = "^X"
        elif has_trends:
            suffix = "^{fd}"
        else:
            suffix = ""

        # For delta, suffix goes after: delta^X, delta^{fd}
        if base == "delta" and suffix:
            return f"delta{suffix}"
        # For DID variants, suffix goes on DID: DID^X_1, DID^{fd}_M
        if suffix:
            did_part = base.split("_")[0]  # "DID"
            sub_part = base.split("_")[1] if "_" in base else ""
            return f"{did_part}{suffix}_{sub_part}" if sub_part else f"{did_part}{suffix}"
        return base

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
        label = self._estimand_label()
        return (
            f"ChaisemartinDHaultfoeuilleResults("
            f"{label}={self.overall_att:.4f}{sig}, "
            f"SE={self.overall_se:.4f}, "
            f"n_groups={len(self.groups)}, "
            f"n_switcher_cells={self.n_switcher_cells})"
        )

    @property
    def coef_var(self) -> float:
        """SE / abs(DID_M); NaN when DID_M is 0 or SE non-finite."""
        if not (np.isfinite(self.overall_se) and self.overall_se >= 0):
            return np.nan
        if not np.isfinite(self.overall_att) or self.overall_att == 0:
            return np.nan
        return self.overall_se / abs(self.overall_att)

    @property
    def is_significant(self) -> bool:
        """True iff overall ``DID_M`` p-value is below ``alpha``."""
        return bool(self.overall_p_value < self.alpha)

    @property
    def significance_stars(self) -> str:
        """Significance stars for the overall ``DID_M``."""
        return _get_significance_stars(self.overall_p_value)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self, alpha: Optional[float] = None) -> str:
        """
        Generate a formatted summary of dCDH estimation results.

        Parameters
        ----------
        alpha : float, optional
            Significance level for the confidence interval header. Defaults
            to ``self.alpha``.

        Returns
        -------
        str
            Formatted multi-block summary including overall ``DID_M``,
            joiners-only / leavers-only views, the placebo, the TWFE
            decomposition diagnostic, and a footer of significance codes.
        """
        alpha = alpha or self.alpha
        conf_level = int((1 - alpha) * 100)
        width = 85
        sep = "=" * width
        thin = "-" * width
        header_row = (
            f"{'Parameter':<15} {'Estimate':>12} {'Std. Err.':>12} "
            f"{'t-stat':>10} {'P>|t|':>10} {'Sig.':>6}"
        )

        lines = [
            sep,
            "de Chaisemartin-D'Haultfoeuille (dCDH) Estimator Results".center(width),
            sep,
            "",
            f"{'Total observations:':<35} {self.n_obs:>10}",
            f"{'Treated observations:':<35} {self.n_treated_obs:>10}",
            (
                f"{'Eligible switchers (N_1):':<35} {self.n_switcher_cells:>10}"
                if self.L_max is not None and self.L_max >= 1
                else f"{'Switcher cells (N_S):':<35} {self.n_switcher_cells:>10}"
            ),
            f"{'Groups (post-filter):':<35} {len(self.groups):>10}",
            f"{'Cohorts:':<35} {self.n_cohorts:>10}",
            f"{'Time periods:':<35} {len(self.time_periods):>10}",
            "",
        ]

        # Filter counts (only show if any drops/exclusions happened).
        # After Round 2, never-switching groups participate in the variance
        # via stable-control roles and are NOT dropped — their count is
        # reported here for backwards compatibility only.
        if (
            self.n_groups_dropped_crossers
            + self.n_groups_dropped_singleton_baseline
            + self.n_groups_dropped_never_switching
            > 0
        ):
            lines.extend(
                [
                    "Group filter / metadata counts:",
                    f"{'  Multi-switch (dropped):':<42} " f"{self.n_groups_dropped_crossers:>10}",
                    f"{'  Singleton baseline (variance only):':<42} "
                    f"{self.n_groups_dropped_singleton_baseline:>10}",
                    f"{'  Never-switching (reported, not dropped):':<42} "
                    f"{self.n_groups_dropped_never_switching:>10}",
                    "",
                ]
            )

        # --- Overall ---
        has_controls = self.covariate_residuals is not None
        has_trends = self._has_trends_linear()
        adj_tag = ""
        if has_controls and has_trends:
            adj_tag = " (Covariate-and-Trend-Adjusted)"
        elif has_controls:
            adj_tag = " (Covariate-Adjusted)"
        elif has_trends:
            adj_tag = " (Trend-Adjusted)"

        if self.L_max is not None and self.L_max >= 2:
            if has_trends:
                overall_label = f"Overall (N/A under trends_linear){adj_tag}"
            else:
                overall_label = f"Cost-Benefit Delta{adj_tag}"
            overall_row_label = self._estimand_label()
        elif self.L_max is not None and self.L_max == 1:
            overall_label = f"Per-Group ATT at Horizon 1{adj_tag}"
            overall_row_label = self._estimand_label()
        else:
            overall_label = f"DID_M (Contemporaneous-Switch ATT){adj_tag}"
            overall_row_label = self._estimand_label()
        lines.extend(
            [
                thin,
                overall_label.center(width),
                thin,
                header_row,
                thin,
                _format_inference_row(
                    overall_row_label,
                    self.overall_att,
                    self.overall_se,
                    self.overall_t_stat,
                    self.overall_p_value,
                ),
                thin,
                "",
                f"{conf_level}% Confidence Interval: "
                f"[{_fmt_float(self.overall_conf_int[0])}, "
                f"{_fmt_float(self.overall_conf_int[1])}]",
            ]
        )

        cv = self.coef_var
        if np.isfinite(cv):
            cv_label = f"CV (SE/abs({overall_row_label})):"
            lines.append(f"{cv_label:<25} {cv:>10.4f}")

        lines.append("")
        is_delta = (
            self.L_max is not None and self.L_max >= 2 and self.cost_benefit_delta is not None
        )
        # Footer labeling is keyed off what the displayed fields actually
        # are — with the bootstrap-contract NaN-on-invalid fix on the fit()
        # side, ``self.overall_se`` is NaN when the bootstrap produced
        # non-finite output, so the "multiplier-bootstrap percentile
        # inference" claim correctly fires only when the displayed overall
        # SE / p-value / CI were actually populated from finite bootstrap
        # output. The event-study fallback branch also checks that at
        # least one horizon has a finite SE before claiming bootstrap was
        # "used for event-study horizon inference" — otherwise every
        # bootstrap inference field is NaN and we fall through to the
        # "bootstrap attempted but invalid" note. The `any_finite_*`
        # predicate below expands the check to cover joiners / leavers /
        # path_effects too: by_path zeros switcher contributions for non-
        # path groups while keeping controls intact, so a per-path
        # bootstrap target can produce a finite SE even when the overall
        # / event-study bootstrap is degenerate (e.g., a reversible panel
        # where the overall mix of joiners + leavers produces a zero
        # centered IF while individual paths do not). Without this
        # broader predicate, the footer would falsely claim "produced
        # non-finite SE on every target" while a finite per-path
        # bootstrap SE sits in the rendered output below.
        event_study_has_finite_bootstrap_se = self.event_study_effects is not None and any(
            np.isfinite(entry.get("se", np.nan)) for entry in self.event_study_effects.values()
        )
        joiners_has_finite_bootstrap_se = self.joiners_se is not None and np.isfinite(
            self.joiners_se
        )
        leavers_has_finite_bootstrap_se = self.leavers_se is not None and np.isfinite(
            self.leavers_se
        )
        path_effects_has_finite_bootstrap_se = self.path_effects is not None and any(
            np.isfinite(h.get("se", np.nan))
            for entry in self.path_effects.values()
            for h in entry.get("horizons", {}).values()
        )
        path_placebo_has_finite_bootstrap_se = self.path_placebo_event_study is not None and any(
            np.isfinite(h.get("se", np.nan))
            for entry in self.path_placebo_event_study.values()
            for h in entry.values()
        )
        path_sup_t_has_finite_crit = self.path_sup_t_bands is not None and any(
            np.isfinite(v.get("crit_value", np.nan)) for v in self.path_sup_t_bands.values()
        )
        any_finite_bootstrap_inference = (
            np.isfinite(self.overall_se)
            or event_study_has_finite_bootstrap_se
            or joiners_has_finite_bootstrap_se
            or leavers_has_finite_bootstrap_se
            or path_effects_has_finite_bootstrap_se
            or path_placebo_has_finite_bootstrap_se
            or path_sup_t_has_finite_crit
        )
        if self.bootstrap_results is not None and np.isfinite(self.overall_se) and not is_delta:
            lines.append("Note: p-value and CI are multiplier-bootstrap percentile inference")
            lines.append(
                f"      ({self.bootstrap_results.n_bootstrap} iterations, "
                f"{self.bootstrap_results.weight_type} weights)."
            )
        elif (
            self.bootstrap_results is not None and is_delta and event_study_has_finite_bootstrap_se
        ):
            lines.append(
                f"Note: delta SE is delta-method (normal-theory) from per-horizon "
                f"bootstrap SEs ({self.bootstrap_results.n_bootstrap} iterations)."
            )
        elif self.bootstrap_results is not None and event_study_has_finite_bootstrap_se:
            lines.append(
                f"Note: bootstrap ({self.bootstrap_results.n_bootstrap} iterations) "
                f"used for event-study horizon inference."
            )
        elif self.bootstrap_results is not None and any_finite_bootstrap_inference:
            # Overall / event-study degenerated but joiners / leavers /
            # path_effects still have finite bootstrap SE. Point the reader
            # at the targets that succeeded rather than claiming a blanket
            # failure.
            live_targets = []
            if joiners_has_finite_bootstrap_se:
                live_targets.append("joiners")
            if leavers_has_finite_bootstrap_se:
                live_targets.append("leavers")
            if path_effects_has_finite_bootstrap_se:
                live_targets.append("per-path")
            if path_placebo_has_finite_bootstrap_se:
                live_targets.append("per-path placebo")
            if path_sup_t_has_finite_crit:
                live_targets.append("per-path sup-t")
            lines.append(
                f"Note: bootstrap ({self.bootstrap_results.n_bootstrap} iterations) "
                f"produced non-finite SE on the overall/event-study target; "
                f"{', '.join(live_targets)} bootstrap inference is populated."
            )
        elif self.bootstrap_results is not None:
            lines.append(
                f"Note: bootstrap ({self.bootstrap_results.n_bootstrap} iterations) "
                f"was requested but produced non-finite SE on every target; all "
                f"inference fields are NaN-consistent per the bootstrap contract."
            )
        else:
            lines.append(
                "Note: dCDH analytical CI is conservative under Assumption 8"
                " (independent groups);"
            )
            lines.append("      exact under iid sampling.")
        lines.append("")

        # --- Joiners and leavers ---
        lines.extend(
            [
                thin,
                "Decomposition: Joiners (DID_+) and Leavers (DID_-)".center(width),
                thin,
                header_row,
                thin,
            ]
        )

        if self.joiners_available:
            lines.append(
                _format_inference_row(
                    "DID_+",
                    self.joiners_att,
                    self.joiners_se,
                    self.joiners_t_stat,
                    self.joiners_p_value,
                )
            )
            lines.append(
                f"  ({self.n_joiner_cells} joiner cells, " f"{self.n_joiner_obs} observations)"
            )
        else:
            lines.append(
                f"{'DID_+':<15} {'(no joiners)':>12} " f"{'':>12} {'':>10} {'':>10} {'':>6}"
            )

        if self.leavers_available:
            lines.append(
                _format_inference_row(
                    "DID_-",
                    self.leavers_att,
                    self.leavers_se,
                    self.leavers_t_stat,
                    self.leavers_p_value,
                )
            )
            lines.append(
                f"  ({self.n_leaver_cells} leaver cells, " f"{self.n_leaver_obs} observations)"
            )
        else:
            lines.append(
                f"{'DID_-':<15} {'(no leavers)':>12} " f"{'':>12} {'':>10} {'':>10} {'':>6}"
            )

        lines.extend([thin, ""])

        # --- Placebo ---
        if self.placebo_available:
            lines.extend(
                [
                    thin,
                    "Single-Lag Placebo (DID_M^pl)".center(width),
                    thin,
                    header_row,
                    thin,
                    _format_inference_row(
                        "DID_M^pl",
                        self.placebo_effect,
                        self.placebo_se,
                        self.placebo_t_stat,
                        self.placebo_p_value,
                    ),
                    thin,
                    "Under parallel trends, the placebo should be ~0.",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    thin,
                    "Placebo not available (T < 3 or no qualifying cells)".center(width),
                    thin,
                    "",
                ]
            )

        # --- Event study table (L_max >= 1) ---
        if self.L_max is not None and self.L_max >= 1 and self.event_study_effects:
            lines.extend(
                [
                    thin,
                    f"Event Study ({self._horizon_label('l')}, l = 1..{self.L_max})".center(width),
                    thin,
                    header_row,
                    thin,
                ]
            )
            for l_h in sorted(self.event_study_effects.keys()):
                entry = self.event_study_effects[l_h]
                lines.append(
                    _format_inference_row(
                        self._horizon_label(l_h),
                        entry["effect"],
                        entry["se"],
                        entry["t_stat"],
                        entry["p_value"],
                    )
                )
            lines.extend([thin])

            # Sup-t bands note
            if self.sup_t_bands is not None:
                crit = self.sup_t_bands["crit_value"]
                lines.append(
                    f"Sup-t critical value: {crit:.4f} " f"(simultaneous {conf_level}% bands)"
                )

            # Cost-benefit delta
            if self.cost_benefit_delta is not None:
                delta = self.cost_benefit_delta.get("delta", float("nan"))
                lines.extend(
                    [
                        "",
                        f"{'Cost-benefit delta:':<35} {_fmt_float(delta):>10}",
                    ]
                )
                if self.cost_benefit_delta.get("has_leavers", False):
                    dj = self.cost_benefit_delta.get("delta_joiners", float("nan"))
                    dl = self.cost_benefit_delta.get("delta_leavers", float("nan"))
                    lines.append(
                        f"  (Assumption 7 violated: joiners={_fmt_float(dj)}, "
                        f"leavers={_fmt_float(dl)})"
                    )

            # Dynamic placebos
            if self.placebo_event_study:
                lines.extend(
                    [
                        "",
                        f"{'Placebos:':<15}",
                    ]
                )
                for h in sorted(self.placebo_event_study.keys()):
                    entry = self.placebo_event_study[h]
                    eff = _fmt_float(entry["effect"])
                    n_pl = entry["n_obs"]
                    lines.append(f"  DID^pl_{abs(h)}: {eff:>10}  (N={n_pl})")

            lines.extend([""])

        # --- Phase 3 extension blocks (factored into helpers) ---
        self._render_covariate_section(lines, width, thin)
        self._render_linear_trends_section(lines, width, thin, header_row)
        self._render_heterogeneity_section(lines, width, thin)
        self._render_design2_section(lines, width, thin)
        self._render_path_effects_section(lines, width, thin, header_row)
        self._render_honest_did_section(lines, width, thin)

        # --- TWFE diagnostic ---
        if self.twfe_beta_fe is not None:
            lines.extend(
                [
                    thin,
                    "TWFE Decomposition Diagnostic (Theorem 1, AER 2020)".center(width),
                    thin,
                    f"{'Plain TWFE coefficient:':<35} {_fmt_float(self.twfe_beta_fe):>10}",
                ]
            )
            if self.twfe_fraction_negative is not None:
                lines.append(
                    f"{'Fraction of negative weights:':<35} "
                    f"{self.twfe_fraction_negative:>10.4f}"
                )
            if self.twfe_sigma_fe is not None and np.isfinite(self.twfe_sigma_fe):
                lines.append(
                    f"{'Sigma_fe (sign-flip threshold):':<35} " f"{self.twfe_sigma_fe:>10.4f}"
                )
            lines.extend(
                [
                    "",
                    "A positive fraction of negative weights signals that the plain",
                    "TWFE coefficient may have the wrong sign under heterogeneous",
                    "treatment effects. Sigma_fe is the smallest cell-level effect",
                    "standard deviation that could flip the sign of TWFE.",
                    thin,
                    "",
                ]
            )

        lines.extend(
            [
                "Signif. codes: '***' 0.001, '**' 0.01, '*' 0.05, '.' 0.1",
                sep,
            ]
        )

        return "\n".join(lines)

    def print_summary(self, alpha: Optional[float] = None) -> None:
        """Print the formatted summary to stdout."""
        print(self.summary(alpha))

    # ------------------------------------------------------------------
    # Summary section helpers (Phase 3 blocks)
    # ------------------------------------------------------------------

    def _render_covariate_section(self, lines: List[str], width: int, thin: str) -> None:
        if self.covariate_residuals is None:
            return
        cov_df = self.covariate_residuals
        control_names = sorted(cov_df["covariate"].unique())
        n_baselines = cov_df["baseline_treatment"].nunique()
        failed = int((cov_df.groupby("baseline_treatment")["theta_hat"].first().isna()).sum())
        lines.extend(
            [
                thin,
                "Covariate Adjustment (DID^X) Diagnostics".center(width),
                thin,
                f"{'Controls:':<35} {', '.join(control_names):>10}",
                f"{'Baselines residualized:':<35} {n_baselines:>10}",
                f"{'Failed strata:':<35} {failed:>10}",
                thin,
                "",
            ]
        )

    def _render_linear_trends_section(
        self, lines: List[str], width: int, thin: str, header_row: str
    ) -> None:
        if self.linear_trends_effects is None:
            return
        lines.extend(
            [
                thin,
                "Cumulated Level Effects (DID^{fd}, trends_linear)".center(width),
                thin,
                header_row,
                thin,
            ]
        )
        for l_h in sorted(self.linear_trends_effects.keys()):
            entry = self.linear_trends_effects[l_h]
            lines.append(
                _format_inference_row(
                    f"Level_{l_h}",
                    entry["effect"],
                    entry["se"],
                    entry["t_stat"],
                    entry["p_value"],
                )
            )
        lines.extend([thin, ""])

    def _render_heterogeneity_section(self, lines: List[str], width: int, thin: str) -> None:
        if self.heterogeneity_effects is None:
            return
        lines.extend(
            [
                thin,
                "Heterogeneity Test (Section 1.5, partial)".center(width),
                thin,
                f"{'Horizon':<15} {'beta^het':>12} {'Std. Err.':>12} "
                f"{'t-stat':>10} {'P>|t|':>10} {'Sig.':>6}",
                thin,
            ]
        )
        for l_h in sorted(self.heterogeneity_effects.keys()):
            entry = self.heterogeneity_effects[l_h]
            lines.append(
                _format_inference_row(
                    f"l={l_h}",
                    entry["beta"],
                    entry["se"],
                    entry["t_stat"],
                    entry["p_value"],
                )
            )
        lines.extend(
            [
                thin,
                "Note: Per-horizon regressions only (no joint F-test). "
                "Negative l = placebo (backward) horizon when "
                "placebo=True. Under survey_design, only forward "
                "horizons are computed (backward-horizon survey "
                "heterogeneity is deferred — see REGISTRY note).",
                "",
            ]
        )

    def _render_design2_section(self, lines: List[str], width: int, thin: str) -> None:
        if self.design2_effects is None:
            return
        d2 = self.design2_effects
        si = d2.get("switch_in", {})
        so = d2.get("switch_out", {})
        lines.extend(
            [
                thin,
                "Design-2: Switch-In / Switch-Out (Section 1.6)".center(width),
                thin,
                f"{'Join-then-leave groups:':<35} {d2.get('n_design2_groups', 0):>10}",
                f"{'Switch-in effect (mean):':<35} "
                f"{_fmt_float(si.get('mean_effect', float('nan'))):>10}"
                f"  (N={si.get('n_groups', 0)})",
                f"{'Switch-out effect (mean):':<35} "
                f"{_fmt_float(so.get('mean_effect', float('nan'))):>10}"
                f"  (N={so.get('n_groups', 0)})",
                thin,
                "",
            ]
        )

    def _render_path_effects_section(
        self, lines: List[str], width: int, thin: str, header_row: str
    ) -> None:
        # Distinguish "by_path not requested" (None) from "requested but
        # empty" ({}). On empty, render a notice so the user sees the
        # feature was active but produced no rows.
        if self.path_effects is None:
            return
        if not self.path_effects:
            # Distinguish the two empty causes for paths_of_interest
            # users (every requested path unobserved) from by_path=k
            # users (no panel path has a complete window).
            poi = getattr(self._estimator_ref, "paths_of_interest", None)
            if poi is not None:
                detail_lines = [
                    "  Every path in paths_of_interest was unobserved or had a window outside L_max+1.",
                    "  (See per-path 'zero observed groups' UserWarnings emitted at fit().)",
                ]
            else:
                detail_lines = [
                    "  No observed paths have a complete [F_g-1, F_g-1+L_max] window.",
                    "  (See UserWarning emitted at fit(); by_path was a no-op on this panel.)",
                ]
            lines.extend(
                [
                    thin,
                    "Treatment-Path Disaggregation".center(width),
                    thin,
                    *detail_lines,
                    thin,
                    "",
                ]
            )
            return
        lines.extend(
            [
                thin,
                "Treatment-Path Disaggregation".center(width),
                thin,
            ]
        )
        # Iterate in path_effects insertion order so summary preserves
        # the user-specified path order under `paths_of_interest`. Under
        # `by_path=k`, insertion order matches descending frequency_rank
        # (the enumeration sorts by count), so the rendering is identical.
        for path in self.path_effects.keys():
            entry = self.path_effects[path]
            rank = entry["frequency_rank"]
            n_groups = entry["n_groups"]
            path_label = f"Path {path}"
            lines.extend(
                [
                    f"  Rank #{rank}: {path_label}  (n_groups={n_groups})",
                    header_row,
                    thin,
                ]
            )
            horizons = entry.get("horizons", {})
            # Backward placebo lags first (negative-keyed), then
            # positive event-study horizons. Skips silently when
            # path_placebo_event_study is None or this path lacks an
            # entry.
            placebo_horizons = (
                self.path_placebo_event_study.get(path, {})
                if self.path_placebo_event_study is not None
                else {}
            )
            for lag_key in sorted(placebo_horizons.keys()):
                ph = placebo_horizons[lag_key]
                lines.append(
                    _format_inference_row(
                        f"  l={lag_key}",
                        ph["effect"],
                        ph["se"],
                        ph["t_stat"],
                        ph["p_value"],
                    )
                )
            for l_h in sorted(horizons.keys()):
                h = horizons[l_h]
                lines.append(
                    _format_inference_row(
                        f"  l={l_h}",
                        h["effect"],
                        h["se"],
                        h["t_stat"],
                        h["p_value"],
                    )
                )
            # Per-path cumulated level effects (under trends_linear).
            # Mirrors the global linear_trends_effects rendering inside
            # the per-path block: appears as a labeled sub-block right
            # after the per-horizon DID^{fd}_l rows. Skip silently when
            # path_cumulated_event_study is None or this path lacks an
            # entry (the latter shouldn't happen, but kept as a guard).
            if (
                self.path_cumulated_event_study is not None
                and path in self.path_cumulated_event_study
            ):
                cum_horizons = self.path_cumulated_event_study[path]
                if cum_horizons:
                    lines.append("  Cumulated Level Effects (DID^{fd}, trends_linear):")
                    for l_h in sorted(cum_horizons.keys()):
                        ce = cum_horizons[l_h]
                        lines.append(
                            _format_inference_row(
                                f"  Level_{l_h}",
                                ce["effect"],
                                ce["se"],
                                ce["t_stat"],
                                ce["p_value"],
                            )
                        )
            # Per-path heterogeneity rows (under heterogeneity=col).
            # Mirrors the global `_render_heterogeneity_section` block
            # but scoped to this path. Skip silently when
            # path_heterogeneity_effects is None or this path lacks an
            # entry (e.g., when `heterogeneity` was not requested).
            if (
                self.path_heterogeneity_effects is not None
                and path in self.path_heterogeneity_effects
            ):
                het_horizons = self.path_heterogeneity_effects[path]
                if het_horizons:
                    lines.append("  Heterogeneity Test (Section 1.5, partial):")
                    for l_h in sorted(het_horizons.keys()):
                        het = het_horizons[l_h]
                        lines.append(
                            _format_inference_row(
                                f"  l={l_h}",
                                het["beta"],
                                het["se"],
                                het["t_stat"],
                                het["p_value"],
                            )
                        )
            # Per-path joint sup-t critical value (when populated).
            # Mirrors the OVERALL sup-t crit print at line ~1019.
            if self.path_sup_t_bands is not None and path in self.path_sup_t_bands:
                crit_p = self.path_sup_t_bands[path].get("crit_value", np.nan)
                if np.isfinite(crit_p):
                    conf_level = int((1 - self.alpha) * 100)
                    lines.append(
                        f"  Sup-t critical value: {crit_p:.4f} "
                        f"(simultaneous {conf_level}% bands)"
                    )
            lines.extend([thin])
        lines.extend([""])

    def _render_honest_did_section(self, lines: List[str], width: int, thin: str) -> None:
        if self.honest_did_results is None:
            return
        hd = self.honest_did_results
        method_label = hd.method.replace("_", " ").title()
        m_val = hd.M
        sig_label = "Yes" if hd.is_significant else "No"
        conf_pct = int((1 - hd.alpha) * 100)
        lines.extend(
            [
                thin,
                "HonestDiD Sensitivity (Rambachan-Roth 2023)".center(width),
                thin,
                f"{'Method:':<35} {method_label} (M={_fmt_float(m_val)})",
                f"{'Target:':<35} {hd.target_label}",
            ]
        )
        if hd.post_periods_used is not None:
            lines.append(f"{'Post horizons used:':<35} {hd.post_periods_used}")
        if hd.pre_periods_used is not None:
            lines.append(f"{'Pre horizons used:':<35} {hd.pre_periods_used}")
        lines.extend(
            [
                f"{'Original estimate:':<35} {_fmt_float(hd.original_estimate):>10}",
                f"{'Identified set:':<35} " f"[{_fmt_float(hd.lb)}, {_fmt_float(hd.ub)}]",
                f"{'Robust ' + str(conf_pct) + '% CI:':<35} "
                f"[{_fmt_float(hd.ci_lb)}, {_fmt_float(hd.ci_ub)}]",
                f"{'Significant at ' + str(int(hd.alpha * 100)) + '%:':<35} " f"{sig_label:>10}",
                thin,
                "",
            ]
        )

    # ------------------------------------------------------------------
    # to_dataframe
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert headline results to a dictionary.

        Returns
        -------
        Dict[str, Any]
            Canonical inference row plus scalar metadata; joiner / leaver /
            placebo decompositions are included only when available.
            Detailed tables are available via ``to_dataframe(level=...)``.
        """
        result = {
            "att": self.att,
            "se": self.se,
            "t_stat": self.t_stat,
            "p_value": self.p_value,
            "conf_int_lower": self.overall_conf_int[0],
            "conf_int_upper": self.overall_conf_int[1],
            "n_obs": self.n_obs,
            "n_treated_obs": self.n_treated_obs,
            "n_switcher_cells": self.n_switcher_cells,
            "n_cohorts": self.n_cohorts,
            "L_max": self.L_max,
            "alpha": self.alpha,
        }
        if self.joiners_available:
            result["joiners_att"] = self.joiners_att
            result["joiners_se"] = self.joiners_se
        if self.leavers_available:
            result["leavers_att"] = self.leavers_att
            result["leavers_se"] = self.leavers_se
        if self.placebo_available:
            result["placebo_effect"] = self.placebo_effect
            result["placebo_se"] = self.placebo_se
        return result

    def to_dataframe(self, level: str = "overall") -> pd.DataFrame:
        """
        Convert results to a DataFrame at the requested level of aggregation.

        Parameters
        ----------
        level : str, default="overall"
            One of:

            - ``"overall"``: single-row table with the overall estimand
              (``DID_M`` when ``L_max=None``, ``DID_1`` when ``L_max=1``,
              ``delta`` when ``L_max >= 2``).
            - ``"joiners_leavers"``: up to three rows for the overall,
              ``DID_+``, and ``DID_-`` (binary panels only).
            - ``"per_period"``: one row per time period with
              ``did_plus_t``, ``did_minus_t``, switching cell counts, and
              the A11-zeroed flags.
            - ``"event_study"``: one row per horizon (positive and
              negative/placebo), including a reference period at
              horizon 0. Available when ``L_max >= 1``.
            - ``"normalized"``: one row per horizon for the normalized
              effects ``DID^n_l``. Available when ``L_max >= 1``.
            - ``"twfe_weights"``: per-(group, time) TWFE decomposition
              weights table. Only available when ``twfe_diagnostic=True``
              was passed to ``fit()``.
            - ``"heterogeneity"``: one row per horizon for the
              heterogeneity test ``beta^{het}_l``. Available when
              ``heterogeneity`` is passed to ``fit()``.
            - ``"linear_trends"``: one row per horizon for the
              cumulated trend-adjusted level effects ``delta^{fd}_l``.
              Available when ``trends_linear=True``.
            - ``"design2"``: Design-2 switch-in/switch-out descriptive
              summary. Available when ``design2=True``.
            - ``"by_path"``: one row per (path, horizon) when either
              ``by_path=k`` or ``paths_of_interest=[(...), ...]`` was
              passed to the estimator. Columns:
              ``path``, ``frequency_rank``, ``n_groups``, ``horizon``,
              ``effect``, ``se``, ``t_stat``, ``p_value``,
              ``conf_int_lower``, ``conf_int_upper``, ``n_obs``,
              ``cband_lower``, ``cband_upper``, ``cumulated_effect``,
              ``cumulated_se``, ``het_beta``, ``het_se``,
              ``het_t_stat``, ``het_p_value``, ``het_conf_int_lower``,
              ``het_conf_int_upper``. The ``horizon`` column takes
              negative ints for placebo rows when ``placebo=True``. The
              ``cband_*`` columns mirror the OVERALL
              ``level="event_study"`` schema (joint sup-t simultaneous
              bands); they are populated for positive-horizon rows of
              paths with a finite per-path sup-t crit (``n_bootstrap >
              0``) and NaN otherwise (placebo rows, unbanded paths, or
              the requested-but-empty fallback DataFrame). The
              ``cumulated_*`` columns mirror the global
              ``linear_trends_effects`` cumulation; populated for
              positive-horizon rows when ``trends_linear=True`` is
              also set, NaN for placebo rows or non-trends_linear fits
              (always-present, NaN-when-None — same convention as
              ``cband_*``). The ``het_*`` columns surface the per-path
              heterogeneity coefficient (Web Appendix Section 1.5,
              Lemma 7) when ``heterogeneity="<col>"`` is also set.
              Populated for positive-horizon (forward) rows whenever
              heterogeneity is requested, AND for negative-horizon
              (placebo) rows when ``placebo=True`` is also set
              (post-2026-05-15: per-path placebo predict_het R-parity
              against ``did_multiplegt_dyn(by_path, predict_het, placebo)``).
              NaN for non-heterogeneity fits / the requested-but-empty
              fallback DataFrame, AND for placebo rows under
              ``survey_design`` (forward-only fallback — backward-horizon
              survey predict_het is deferred until the pre-period cell
              allocator is derived; a ``UserWarning`` fires at fit-time
              when ``survey_design + placebo + heterogeneity`` are
              co-set). Always-present, NaN-when-None — same convention
              as ``cband_*`` and ``cumulated_*``.

        Returns
        -------
        pd.DataFrame
        """
        if level == "overall":
            return pd.DataFrame(
                [
                    {
                        "estimand": self._estimand_label(),
                        "effect": self.overall_att,
                        "se": self.overall_se,
                        "t_stat": self.overall_t_stat,
                        "p_value": self.overall_p_value,
                        "conf_int_lower": self.overall_conf_int[0],
                        "conf_int_upper": self.overall_conf_int[1],
                    }
                ]
            )

        elif level == "joiners_leavers":
            # Two separate count columns so each has consistent units
            # across all rows:
            #   n_cells: total switching cells (each (g, t) cell counted once)
            #   n_obs:   actual observation count summed over the same cells
            #            (equals n_cells on balanced 1-obs-per-cell panels;
            #            larger on individual-level inputs with multiple
            #            observations per cell).
            # For the DID_M row, both quantities use the overall switching
            # cell set: n_cells = sum of joiner + leaver cells, and n_obs
            # is the same sum of raw observation counts.
            overall_est_label = self._estimand_label()
            rows = [
                {
                    "estimand": overall_est_label,
                    "effect": self.overall_att,
                    "se": self.overall_se,
                    "t_stat": self.overall_t_stat,
                    "p_value": self.overall_p_value,
                    "conf_int_lower": self.overall_conf_int[0],
                    "conf_int_upper": self.overall_conf_int[1],
                    "n_cells": self.n_switcher_cells,
                    "n_obs": (
                        self.n_treated_obs
                        if not self.joiners_available and not self.leavers_available
                        else self.n_joiner_obs + self.n_leaver_obs
                    ),
                    "available": True,
                },
                {
                    "estimand": "DID_+",
                    "effect": self.joiners_att,
                    "se": self.joiners_se,
                    "t_stat": self.joiners_t_stat,
                    "p_value": self.joiners_p_value,
                    "conf_int_lower": self.joiners_conf_int[0],
                    "conf_int_upper": self.joiners_conf_int[1],
                    "n_cells": self.n_joiner_cells,
                    "n_obs": self.n_joiner_obs,
                    "available": self.joiners_available,
                },
                {
                    "estimand": "DID_-",
                    "effect": self.leavers_att,
                    "se": self.leavers_se,
                    "t_stat": self.leavers_t_stat,
                    "p_value": self.leavers_p_value,
                    "conf_int_lower": self.leavers_conf_int[0],
                    "conf_int_upper": self.leavers_conf_int[1],
                    "n_cells": self.n_leaver_cells,
                    "n_obs": self.n_leaver_obs,
                    "available": self.leavers_available,
                },
            ]
            return pd.DataFrame(rows)

        elif level == "per_period":
            if not self.per_period_effects:
                # Empty per-period table — return DataFrame with the
                # canonical column order so downstream code can rely on it.
                return pd.DataFrame(
                    {
                        "period": pd.Series(dtype="int64"),
                        "did_plus_t": pd.Series(dtype="float64"),
                        "did_minus_t": pd.Series(dtype="float64"),
                        "n_10_t": pd.Series(dtype="int64"),
                        "n_01_t": pd.Series(dtype="int64"),
                        "n_00_t": pd.Series(dtype="int64"),
                        "n_11_t": pd.Series(dtype="int64"),
                        "did_plus_t_a11_zeroed": pd.Series(dtype="bool"),
                        "did_minus_t_a11_zeroed": pd.Series(dtype="bool"),
                    }
                )
            rows = []
            for t in sorted(self.per_period_effects.keys()):
                cell = self.per_period_effects[t]
                rows.append({"period": t, **cell})
            return pd.DataFrame(rows)

        elif level == "event_study":
            rows = []
            # Placebo horizons (negative keys)
            if self.placebo_event_study:
                for h in sorted(self.placebo_event_study.keys()):
                    entry = self.placebo_event_study[h]
                    cband = entry.get("cband_conf_int", (np.nan, np.nan))
                    rows.append(
                        {
                            "horizon": h,
                            "estimand": f"DID^pl_{abs(h)}",
                            "effect": entry["effect"],
                            "se": entry["se"],
                            "t_stat": entry["t_stat"],
                            "p_value": entry["p_value"],
                            "conf_int_lower": entry["conf_int"][0],
                            "conf_int_upper": entry["conf_int"][1],
                            "n_obs": entry["n_obs"],
                            "cband_lower": cband[0] if cband else np.nan,
                            "cband_upper": cband[1] if cband else np.nan,
                        }
                    )
            # Reference period (horizon 0)
            rows.append(
                {
                    "horizon": 0,
                    "estimand": "ref",
                    "effect": 0.0,
                    "se": np.nan,
                    "t_stat": np.nan,
                    "p_value": np.nan,
                    "conf_int_lower": np.nan,
                    "conf_int_upper": np.nan,
                    "n_obs": 0,
                    "cband_lower": np.nan,
                    "cband_upper": np.nan,
                }
            )
            # Positive horizons
            if self.event_study_effects:
                for h in sorted(self.event_study_effects.keys()):
                    entry = self.event_study_effects[h]
                    cband = entry.get("cband_conf_int", (np.nan, np.nan))
                    rows.append(
                        {
                            "horizon": h,
                            "estimand": self._horizon_label(h),
                            "effect": entry["effect"],
                            "se": entry["se"],
                            "t_stat": entry["t_stat"],
                            "p_value": entry["p_value"],
                            "conf_int_lower": entry["conf_int"][0],
                            "conf_int_upper": entry["conf_int"][1],
                            "n_obs": entry["n_obs"],
                            "cband_lower": cband[0] if cband else np.nan,
                            "cband_upper": cband[1] if cband else np.nan,
                        }
                    )
            return pd.DataFrame(rows)

        elif level == "normalized":
            if not self.normalized_effects:
                raise ValueError("Normalized effects not computed. Pass L_max >= 1 to fit().")
            rows = []
            for h in sorted(self.normalized_effects.keys()):
                entry = self.normalized_effects[h]
                rows.append(
                    {
                        "horizon": h,
                        "estimand": f"DID^n_{h}",
                        "effect": entry["effect"],
                        "se": entry["se"],
                        "t_stat": entry["t_stat"],
                        "p_value": entry["p_value"],
                        "conf_int_lower": entry["conf_int"][0],
                        "conf_int_upper": entry["conf_int"][1],
                        "denominator": entry["denominator"],
                    }
                )
            return pd.DataFrame(rows)

        elif level == "twfe_weights":
            if self.twfe_weights is None:
                raise ValueError(
                    "TWFE decomposition weights not computed. Pass "
                    "twfe_diagnostic=True (the default) to ChaisemartinDHaultfoeuille()."
                )
            return self.twfe_weights.copy()

        elif level == "heterogeneity":
            if self.heterogeneity_effects is None:
                raise ValueError(
                    "Heterogeneity test results not available. Pass "
                    "heterogeneity='column_name' to fit()."
                )
            rows = []
            for h, data in sorted(self.heterogeneity_effects.items()):
                rows.append({"horizon": h, **data})
            return pd.DataFrame(rows)

        elif level == "linear_trends":
            if self.linear_trends_effects is None:
                # PR #347 R12 P1: distinguish the "trends_linear was
                # not requested" case from the "trends_linear was
                # requested but no horizons survived" case. Telling
                # a user who already passed ``trends_linear=True``
                # to pass it again is a dead-end.
                if self._has_trends_linear():
                    return pd.DataFrame(
                        columns=[
                            "horizon",
                            "effect",
                            "se",
                            "t_stat",
                            "p_value",
                            "conf_int",
                        ]
                    )
                raise ValueError(
                    "Linear trends effects not available. Pass trends_linear=True to fit()."
                )
            rows = []
            for h, data in sorted(self.linear_trends_effects.items()):
                rows.append({"horizon": h, **data})
            return pd.DataFrame(rows)

        elif level == "design2":
            if self.design2_effects is None:
                raise ValueError(
                    "Design-2 effects not available. Pass "
                    "design2=True with drop_larger_lower=False to fit()."
                )
            return pd.DataFrame([self.design2_effects])

        elif level == "by_path":
            # Distinguish "not requested" from "requested but empty" so the
            # caller who passed by_path=k isn't told to "pass by_path=k".
            # Mirrors the linear_trends pattern above.
            if self.path_effects is None:
                raise ValueError(
                    "Path effects not available. Pass by_path=k "
                    "(positive int) or paths_of_interest=[(...), ...] "
                    "to ChaisemartinDHaultfoeuille(drop_larger_lower=False, "
                    "by_path=k) (or paths_of_interest=...) and L_max >= 1 "
                    "to fit()."
                )
            if not self.path_effects:
                return pd.DataFrame(
                    columns=[
                        "path",
                        "frequency_rank",
                        "n_groups",
                        "horizon",
                        "effect",
                        "se",
                        "t_stat",
                        "p_value",
                        "conf_int_lower",
                        "conf_int_upper",
                        "n_obs",
                        "cband_lower",
                        "cband_upper",
                        "cumulated_effect",
                        "cumulated_se",
                        "het_beta",
                        "het_se",
                        "het_t_stat",
                        "het_p_value",
                        "het_conf_int_lower",
                        "het_conf_int_upper",
                    ]
                )
            rows = []
            # Iterate in path_effects insertion order so the long-format
            # table preserves the user-specified path order under
            # `paths_of_interest`. Under `by_path=k`, insertion order
            # matches descending frequency_rank, so output is identical.
            for path in self.path_effects.keys():
                entry = self.path_effects[path]
                rank = entry["frequency_rank"]
                n_groups = entry["n_groups"]
                horizons = entry.get("horizons", {})
                # Backward placebo lags first (negative-keyed), then
                # positive event-study horizons. Both placebo and
                # event-study rows are emitted in a single
                # `level="by_path"` table so callers see the full
                # forward+backward inference per path.
                placebo_horizons = (
                    self.path_placebo_event_study.get(path, {})
                    if self.path_placebo_event_study is not None
                    else {}
                )
                # Per-path cumulated entries (under trends_linear). Always-
                # present, NaN-when-None mirrors the cband_* convention.
                path_cumulated = (
                    self.path_cumulated_event_study.get(path, {})
                    if self.path_cumulated_event_study is not None
                    else {}
                )
                # Per-path heterogeneity entries (under heterogeneity=col).
                # Always-present het_* columns, NaN when not requested or
                # when the path's per-horizon entry is missing.
                path_het = (
                    self.path_heterogeneity_effects.get(path, {})
                    if self.path_heterogeneity_effects is not None
                    else {}
                )
                for lag_key in sorted(placebo_horizons.keys()):
                    ph_entry = placebo_horizons[lag_key]
                    # Placebos do not get joint sup-t bands in this
                    # release (only positive event-study horizons do —
                    # mirrors OVERALL placebo / event-study sup-t
                    # convention). Emit NaN cband columns for schema
                    # parity with the OVERALL level="event_study" table.
                    # Placebo + cumulated is also NaN: there is no per-
                    # path placebo cumulation surface (placebo under
                    # trends_lin returns RAW per-horizon values per R).
                    ph_cband = ph_entry.get("cband_conf_int", (np.nan, np.nan))
                    # Per-path placebo heterogeneity (TODO #422). R-
                    # verified: did_multiplegt_dyn(..., by_path,
                    # predict_het, placebo) emits per-path predict_het
                    # rows on backward (negative) horizons. Negative
                    # `lag_key` indexes into `path_het` to look up the
                    # placebo het entry; absent key (placebo > 0 but
                    # this (path, lag) is rank-deficient or has < 3
                    # eligible groups) -> NaN columns.
                    ph_het_entry = path_het.get(lag_key, {}) if path_het else {}
                    ph_het_ci = ph_het_entry.get("conf_int", (np.nan, np.nan))
                    rows.append(
                        {
                            "path": path,
                            "frequency_rank": rank,
                            "n_groups": n_groups,
                            "horizon": lag_key,
                            "effect": ph_entry["effect"],
                            "se": ph_entry["se"],
                            "t_stat": ph_entry["t_stat"],
                            "p_value": ph_entry["p_value"],
                            "conf_int_lower": ph_entry["conf_int"][0],
                            "conf_int_upper": ph_entry["conf_int"][1],
                            "n_obs": ph_entry["n_obs"],
                            "cband_lower": ph_cband[0] if ph_cband else np.nan,
                            "cband_upper": ph_cband[1] if ph_cband else np.nan,
                            "cumulated_effect": np.nan,
                            "cumulated_se": np.nan,
                            "het_beta": ph_het_entry.get("beta", np.nan),
                            "het_se": ph_het_entry.get("se", np.nan),
                            "het_t_stat": ph_het_entry.get("t_stat", np.nan),
                            "het_p_value": ph_het_entry.get("p_value", np.nan),
                            "het_conf_int_lower": ph_het_ci[0] if ph_het_ci else np.nan,
                            "het_conf_int_upper": ph_het_ci[1] if ph_het_ci else np.nan,
                        }
                    )
                for l_h in sorted(horizons.keys()):
                    h_entry = horizons[l_h]
                    # Per-path joint sup-t band (when populated) mirrors
                    # OVERALL `level="event_study"` cband emission. Absent
                    # key / missing path entry -> NaN columns. Pinned at
                    # `TestByPathSupTBands::test_path_sup_t_to_dataframe_emits_cband_columns`.
                    h_cband = h_entry.get("cband_conf_int", (np.nan, np.nan))
                    cum_entry = path_cumulated.get(l_h, {})
                    het_entry = path_het.get(l_h, {}) if path_het else {}
                    het_ci = het_entry.get("conf_int", (np.nan, np.nan))
                    rows.append(
                        {
                            "path": path,
                            "frequency_rank": rank,
                            "n_groups": n_groups,
                            "horizon": l_h,
                            "effect": h_entry["effect"],
                            "se": h_entry["se"],
                            "t_stat": h_entry["t_stat"],
                            "p_value": h_entry["p_value"],
                            "conf_int_lower": h_entry["conf_int"][0],
                            "conf_int_upper": h_entry["conf_int"][1],
                            "n_obs": h_entry["n_obs"],
                            "cband_lower": h_cband[0] if h_cband else np.nan,
                            "cband_upper": h_cband[1] if h_cband else np.nan,
                            "cumulated_effect": cum_entry.get("effect", np.nan),
                            "cumulated_se": cum_entry.get("se", np.nan),
                            # Per-path heterogeneity (Wave 5 #11). Always-
                            # present, NaN when not requested or when the
                            # entry is missing (mirrors cband_*/cumulated_*
                            # convention).
                            "het_beta": het_entry.get("beta", np.nan),
                            "het_se": het_entry.get("se", np.nan),
                            "het_t_stat": het_entry.get("t_stat", np.nan),
                            "het_p_value": het_entry.get("p_value", np.nan),
                            "het_conf_int_lower": het_ci[0] if het_ci else np.nan,
                            "het_conf_int_upper": het_ci[1] if het_ci else np.nan,
                        }
                    )
            return pd.DataFrame(rows)

        else:
            raise ValueError(
                f"Unknown level: {level!r}. Use 'overall', 'joiners_leavers', "
                f"'per_period', 'event_study', 'normalized', 'twfe_weights', "
                f"'heterogeneity', 'linear_trends', 'design2', or 'by_path'."
            )


# =============================================================================
# Internal formatting helpers
# =============================================================================


def _fmt_float(x: float) -> str:
    """Format a float; render NaN/Inf as the string 'NaN'/'Inf'."""
    if not np.isfinite(x):
        return "NaN" if np.isnan(x) else ("Inf" if x > 0 else "-Inf")
    return f"{x:.4f}"


def _format_inference_row(
    label: str,
    effect: float,
    se: float,
    t_stat: float,
    p_value: float,
) -> str:
    """Format a single inference row for the summary table."""
    e_str = f"{_fmt_float(effect):>12}"
    s_str = f"{_fmt_float(se):>12}"
    t_str = f"{t_stat:>10.3f}" if np.isfinite(t_stat) else f"{'NaN':>10}"
    p_str = f"{p_value:>10.4f}" if np.isfinite(p_value) else f"{'NaN':>10}"
    sig = _get_significance_stars(p_value) if np.isfinite(p_value) else ""
    return f"{label:<15} {e_str} {s_str} {t_str} {p_str} {sig:>6}"
