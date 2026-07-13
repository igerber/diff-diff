"""
Multiplier-bootstrap inference for the de Chaisemartin-D'Haultfoeuille (dCDH)
estimator.

The dCDH papers prescribe only the analytical cohort-recentered plug-in
variance from Web Appendix Section 3.7.3 of the dynamic companion paper.
This module adds an opt-in multiplier bootstrap, clustered at the group
level by default (matching the inference convention used by
``CallawaySantAnna``, ``ImputationDiD``, and ``TwoStageDiD``). Under
``survey_design`` with an explicitly-coarser PSU, the bootstrap switches
to PSU-level Hall-Mammen wild clustering: each PSU draws a single
multiplier and all groups within that PSU share it (see
``_generate_psu_or_group_weights`` and ``_map_for_target`` below, plus
the REGISTRY.md ``ChaisemartinDHaultfoeuille`` Note on survey +
bootstrap). Under the default auto-inject ``psu=group`` each group is
its own PSU and the identity-map fast path reproduces the original
group-level behavior bit-for-bit.

**Cell-level wild PSU bootstrap (within-group-varying PSU):** when a
survey design's PSU varies across the cells of a group, the group-level
PSU map cannot represent per-PSU contributions (a group with cells in
PSUs ``{p1, p2}`` would collapse to the first PSU, under-clustering the
bootstrap). This module's dispatcher (``_psu_varies_within_group``)
detects the regime and switches to a cell-level allocator: each
observation-level ``psi_i`` is attributed to its ``(g, t)`` cell's PSU
via ``psu_codes_per_cell`` (shape ``(n_eligible_groups, n_periods)``,
-1 sentinel for zero-weight cells), and the bootstrap statistic becomes
``theta_r = sum_c multiplier[psu(c)] * u_centered_pp[c] / divisor``
(using the cohort-recentered per-cell IF ``U_centered_per_period``).
Under PSU-within-group-constant the row-sum identity
``sum_{c in g} u_cell[c] == u_centered[g]`` (enforced by
``_cohort_recenter_per_period``) makes the cell-level and group-level
bootstraps statistically equivalent, and the dispatcher routes to the
legacy group-level path for bit-identity with pre-cell-level releases.
Multi-horizon bootstraps draw a single shared ``(n_bootstrap, n_psu)``
PSU-level weight matrix per block and broadcast per-horizon via each
horizon's cell-to-PSU map, so the sup-t simultaneous band remains a
valid joint distribution. The bootstrap is a library extension, not a
paper requirement, and is documented in ``REGISTRY.md``.

The mixin operates on **pre-computed cohort-centered influence-function
values**: the main estimator class computes per-group ``U^G_g`` values
(and, under survey designs, per-``(g, t)``-cell attributions
``U_centered_per_period[g, t]``) during the analytical variance
calculation, recenters them by their cohort means (using the
``(D_{g,1}, F_g, S_g)`` triple), and stores the recentered vector /
tensor. The bootstrap then multiplies this structure by random
multiplier weights (Rademacher / Mammen / Webb) and re-aggregates to
produce a bootstrap distribution per target.
"""

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import numpy as np

from diff_diff.bootstrap_utils import (
    compute_effect_bootstrap_stats as _compute_effect_bootstrap_stats,
)
from diff_diff.bootstrap_utils import (
    generate_bootstrap_weights_batch as _generate_bootstrap_weights_batch,
)
from diff_diff.chaisemartin_dhaultfoeuille_results import DCDHBootstrapResults

__all__ = ["ChaisemartinDHaultfoeuilleBootstrapMixin"]


class ChaisemartinDHaultfoeuilleBootstrapMixin:
    """
    Bootstrap-inference mixin for ``ChaisemartinDHaultfoeuille``.

    Provides a single entry point ``_compute_dcdh_bootstrap`` that takes
    pre-computed centered influence-function values for each estimand
    target (overall ``DID_M``, joiners ``DID_+``, leavers ``DID_-``,
    placebo ``DID_M^pl``) and returns a populated
    :class:`DCDHBootstrapResults`.

    The mixin is pure (no instance state of its own); it only references
    instance attributes from the main class via ``TYPE_CHECKING`` hints.
    """

    # --- Type hints for attributes accessed from the main class ---
    n_bootstrap: int
    bootstrap_weights: str
    alpha: float
    seed: Optional[int]

    if TYPE_CHECKING:  # pragma: no cover

        def _placeholder(self) -> None: ...  # silences mypy "no attributes" warnings

    def _compute_dcdh_bootstrap(
        self,
        n_groups_for_overall: int,
        u_centered_overall: np.ndarray,
        divisor_overall: int,
        original_overall: float,
        joiners_inputs: Optional[Tuple[np.ndarray, int, float, Optional[np.ndarray]]] = None,
        leavers_inputs: Optional[Tuple[np.ndarray, int, float, Optional[np.ndarray]]] = None,
        placebo_inputs: Optional[Tuple[np.ndarray, int, float, Optional[np.ndarray]]] = None,
        # --- Phase 2: multi-horizon inputs ---
        multi_horizon_inputs: Optional[
            Dict[int, Tuple[np.ndarray, int, float, Optional[np.ndarray]]]
        ] = None,
        placebo_horizon_inputs: Optional[
            Dict[int, Tuple[np.ndarray, int, float, Optional[np.ndarray]]]
        ] = None,
        # --- Phase 3: per-path (by_path) bootstrap inputs ---
        # Nested dict keyed by path tuple -> horizon -> 4-tuple of
        # (u_centered_path, n_l_path, effect_path, u_per_period_optional).
        # The 4th slot is always None in the current release (survey +
        # by_path + bootstrap combination is gated out in fit(); cell-
        # level wild PSU bootstrap for per-path targets is a future
        # wave item).
        path_bootstrap_inputs: Optional[
            Dict[
                Tuple[int, ...],
                Dict[int, Tuple[np.ndarray, int, float, Optional[np.ndarray]]],
            ]
        ] = None,
        # --- Phase 3: per-path placebo (by_path + placebo) bootstrap inputs ---
        # Nested dict keyed by path tuple -> positive lag l (l = 1..L_max)
        # -> 4-tuple `(u_centered_pl_path, n_pl_l_path, effect_pl_path,
        # None)`. Sibling of `path_bootstrap_inputs` for backward
        # placebo horizons; same reason for the 4th-slot `None`
        # (survey + by_path + placebo + bootstrap is gated out, no
        # cell-level path needed).
        path_placebo_bootstrap_inputs: Optional[
            Dict[
                Tuple[int, ...],
                Dict[int, Tuple[np.ndarray, int, float, Optional[np.ndarray]]],
            ]
        ] = None,
        # --- Survey: PSU-level bootstrap under survey designs ---
        group_id_to_psu_code: Optional[Dict[Any, int]] = None,
        eligible_group_ids: Optional[np.ndarray] = None,
        # --- Survey: cell-level wild PSU bootstrap ---
        u_per_period_overall: Optional[np.ndarray] = None,
        psu_codes_per_cell: Optional[np.ndarray] = None,
    ) -> DCDHBootstrapResults:
        """
        Compute multiplier-bootstrap inference for all dCDH targets.

        Each target ``T`` is summarized by:

        - a centered influence-function vector of length equal to the
          number of groups contributing to ``T``
        - a re-aggregation **divisor**, which is the *switching-cell*
          count from the Theorem 3 weighting formula (NOT a group
          count). For ``DID_M`` the divisor is ``N_S = sum_t (N_{1,0,t}
          + N_{0,1,t})``; for ``DID_+`` it is ``sum_t N_{1,0,t}``; for
          ``DID_-`` it is ``sum_t N_{0,1,t}``. See REGISTRY.md
          ``ChaisemartinDHaultfoeuille`` for the cell-count weighting
          contract.
        - the original point estimate of ``T`` (used as the centering
          point for the percentile p-value)

        For each target, this method dispatches between two
        allocator paths based on ``psu_codes_per_cell`` (via the
        ``_psu_varies_within_group`` helper):

        **Legacy group-level path** — runs when
        ``psu_codes_per_cell`` is ``None`` or PSU is within-group-
        constant (PSU=group auto-inject, strictly-coarser PSU with
        within-group constancy, or no survey design). For each target:

        1. Generates an ``(n_bootstrap, n_groups_target)`` matrix of
           multiplier weights via
           :func:`~diff_diff.bootstrap_utils.generate_bootstrap_weights_batch`,
           where ``n_groups_target`` is the IF vector length (one
           weight per contributing group). When a coarser PSU map is
           passed, weights are drawn once per PSU and broadcast to
           groups so all groups in the same PSU share a multiplier.
        2. Computes the bootstrap distribution as
           ``W @ u_centered / divisor`` (one bootstrap replicate per
           row), where ``divisor`` is the switching-cell count
           described above. Note: the weight matrix has one column per
           contributing group, but the divisor is a cell count — the
           two are different quantities (groups can contribute to
           multiple cells across periods).
        3. Passes the distribution + the original point estimate through
           :func:`~diff_diff.bootstrap_utils.compute_effect_bootstrap_stats`
           to obtain ``(SE, CI, p_value)``.

        **Cell-level wild PSU path** — runs when PSU varies within
        group (any row of ``psu_codes_per_cell`` has > 1 unique non-
        sentinel code). For each target:

        1. Unrolls the target's ``u_per_period`` tensor and
           ``psu_codes_per_cell`` to 1-D cell-level arrays via
           :func:`_unroll_target_to_cells`, dropping cells with
           sentinel (-1) PSU. Raises ``ValueError`` if cohort-
           recentered mass lands on a sentinel cell (the terminal-
           missingness guard — see REGISTRY.md).
        2. For **single-target** branches (overall, joiners, leavers,
           placebo-horizon-per-l), generates an
           ``(n_bootstrap, n_cells_in_target)`` weight matrix where
           each cell gets the multiplier of its PSU code, via
           ``psu_weights[:, psu_cell]`` broadcast. Bootstrap
           distribution becomes ``W_cell @ u_cell / divisor``.
        3. For the **multi-horizon** block, draws a SINGLE shared
           ``(n_bootstrap, n_psu)`` PSU-level weight matrix once and
           broadcasts per-horizon via each horizon's cell-to-PSU map.
           Preserves the sup-t simultaneous confidence band as a
           valid joint distribution across horizons.

        Under PSU=group, the row-sum identity
        ``sum_{c in g} u_cell[c] == u_centered[g]`` makes the two
        paths statistically equivalent; the dispatcher prefers the
        legacy path in that regime for bit-identity with pre-PR-4
        releases via the identity-map fast path.

        Parameters
        ----------
        n_groups_for_overall : int
            Number of groups contributing to the overall ``DID_M``
            (length of ``u_centered_overall``). Used for shape
            validation and weight-matrix sizing.
        u_centered_overall : np.ndarray
            Cohort-centered per-group influence-function values for
            ``DID_M``. Shape: ``(n_groups_for_overall,)``.
        divisor_overall : int
            Re-aggregation **divisor** for ``DID_M`` — the switching-
            cell count ``N_S = sum_t (N_{1,0,t} + N_{0,1,t})`` from
            Theorem 3 of AER 2020. NOT a group count. For Phase 1
            this is the same value used in the analytical SE plug-in.
        original_overall : float
            The original point estimate of ``DID_M``. Used by
            :func:`compute_effect_bootstrap_stats` for the percentile
            p-value computation.
        joiners_inputs : tuple, optional
            ``(u_centered, divisor, original_effect, u_per_period)``
            4-tuple for the joiners-only ``DID_+`` target. The
            ``divisor`` is the joiner switching-cell total
            ``sum_t N_{1,0,t}`` (NOT the joiner group count).
            ``u_per_period`` is the cohort-recentered per-cell IF
            tensor of shape ``(n_eligible_groups, n_periods)`` (or
            ``None`` when survey is inactive); it is consumed only
            by the cell-level dispatch described below.
        leavers_inputs : tuple, optional
            Same 4-tuple for the leavers-only ``DID_-`` target.
            ``divisor`` is ``sum_t N_{0,1,t}``.
        placebo_inputs : tuple, optional
            Same 4-tuple for the Phase 1 per-period placebo
            ``DID_M^pl``. ``None`` when ``L_max=None`` (per-period
            placebo has no IF). The 4th slot is always ``None`` here
            because the Phase 1 placebo has no per-cell attribution.
        multi_horizon_inputs : dict, optional
            Per-horizon 4-tuples keyed by horizon ``l``. Same layout
            as ``joiners_inputs``; the per-horizon per-cell tensor
            drives the shared-PSU-weights broadcast under the cell-
            level path.
        placebo_horizon_inputs : dict, optional
            Same shape as ``multi_horizon_inputs`` for the Phase 2
            placebo-horizon targets.
        group_id_to_psu_code : dict, optional
            Group-ID → dense PSU code map for the legacy group-level
            bootstrap path. ``None`` when no PSU info is available.
        eligible_group_ids : np.ndarray, optional
            Ordered group IDs matching the ``u_centered_*`` vectors
            for the legacy group-level path.
        u_per_period_overall : np.ndarray, optional
            Cohort-recentered per-cell IF tensor for the overall
            ``DID_M`` target, shape ``(n_eligible_groups, n_periods)``.
            Consumed only by the cell-level path.
        psu_codes_per_cell : np.ndarray, optional
            Per-cell PSU code tensor, shape
            ``(n_eligible_groups, n_periods)``, with ``-1`` sentinel
            for ineligible / zero-weight cells. Used by
            ``_psu_varies_within_group`` to pick between the legacy
            group-level bootstrap path (PSU-within-group-constant,
            bit-identical to pre-PR-4 behavior via the identity-map
            fast path) and the cell-level wild PSU bootstrap
            (within-group-varying PSU, cell-level multipliers and
            shared PSU-level weights across multi-horizon blocks).

        Returns
        -------
        DCDHBootstrapResults
            Populated bootstrap-results dataclass. Fields for unavailable
            targets (joiners / leavers / placebo) are ``None``.
        """
        if self.n_bootstrap <= 0:
            raise ValueError(
                f"_compute_dcdh_bootstrap called with n_bootstrap={self.n_bootstrap}; "
                "must be > 0."
            )
        if u_centered_overall.ndim != 1:
            raise ValueError(
                "u_centered_overall must be a 1-D array of per-group influence "
                f"function values, got shape {u_centered_overall.shape}"
            )
        if u_centered_overall.shape[0] != n_groups_for_overall:
            raise ValueError(
                f"u_centered_overall length ({u_centered_overall.shape[0]}) does not "
                f"match n_groups_for_overall ({n_groups_for_overall})"
            )
        rng = np.random.default_rng(self.seed)

        # PSU label for each bootstrap weight column is derived from
        # the group's ID via `_map_for_target`, not from positional
        # truncation. All current dCDH bootstrap targets use the
        # variance-eligible group ordering (`eligible_group_ids`); if a
        # future target uses a different ordering, add a dedicated
        # group-IDs parameter for it rather than reusing the overall
        # eligible list.

        # Dispatcher for the cell-level wild PSU bootstrap. When PSU
        # varies across cells of a group, a group-level PSU map
        # collapses within-group PSU variation and the bootstrap
        # under-clusters. The cell-level path draws multipliers at
        # PSU granularity and applies them per (g, t) cell via
        # `psu_codes_per_cell`. Under PSU-within-group-constant
        # regimes (including PSU=group and strictly-coarser PSU
        # within-group-constant), the row-sum identity
        # `sum_{c in g} u_cell[c] == u_centered[g]` makes the two
        # paths statistically equivalent, and the dispatcher routes
        # to the legacy path for bit-identity with pre-release
        # behavior. See REGISTRY.md survey + bootstrap contract Note.
        psu_varies = _psu_varies_within_group(psu_codes_per_cell)

        # --- Overall DID_M ---
        # Skip the scalar DID_M bootstrap when divisor_overall <= 0
        # (e.g., pure non-binary panels where N_S=0), but continue
        # to process multi_horizon_inputs and placebo_horizon_inputs.
        if divisor_overall > 0:
            if psu_varies:
                # Contract: when the cell-level path is required
                # (psu_varies=True), the caller MUST provide the
                # target's per-cell IF tensor. Silent fallback to the
                # group-level allocator would under-cluster the
                # bootstrap by collapsing multi-PSU group
                # contributions to one PSU.
                if u_per_period_overall is None:
                    raise ValueError(
                        "Cell-level bootstrap requires "
                        "u_per_period_overall when PSU varies "
                        "within group, got None. Caller must "
                        "thread the cohort-recentered per-cell IF "
                        "tensor (U_centered_pp_overall) from the "
                        "analytical TSL path."
                    )
                u_boot_overall, map_boot_overall = _unroll_target_to_cells(
                    u_per_period_overall,
                    psu_codes_per_cell,
                )
            else:
                u_boot_overall = u_centered_overall
                map_boot_overall = _map_for_target(
                    u_centered_overall.shape[0],
                    group_id_to_psu_code,
                    eligible_group_ids,
                )
            overall_se, overall_ci, overall_p, overall_dist = _bootstrap_one_target(
                u_centered=u_boot_overall,
                divisor=divisor_overall,
                original=original_overall,
                n_bootstrap=self.n_bootstrap,
                weight_type=self.bootstrap_weights,
                alpha=self.alpha,
                rng=rng,
                context="dCDH overall DID_M bootstrap",
                return_distribution=True,
                group_to_psu_map=map_boot_overall,
            )
        else:
            overall_se = np.nan
            overall_ci = (np.nan, np.nan)
            overall_p = np.nan
            overall_dist = None

        results = DCDHBootstrapResults(
            n_bootstrap=self.n_bootstrap,
            weight_type=self.bootstrap_weights,
            alpha=self.alpha,
            overall_se=overall_se,
            overall_ci=overall_ci,
            overall_p_value=overall_p,
            bootstrap_distribution=overall_dist,
        )

        # --- Joiners (DID_+) ---
        if joiners_inputs is not None:
            u_j, n_j, eff_j, u_pp_j = joiners_inputs
            if u_j.size > 0 and n_j > 0:
                if psu_varies:
                    if u_pp_j is None:
                        raise ValueError(
                            "Cell-level bootstrap requires joiners' "
                            "per-cell IF tensor (U_centered_pp_joiners) "
                            "when PSU varies within group, got None."
                        )
                    u_boot_j, map_boot_j = _unroll_target_to_cells(
                        u_pp_j,
                        psu_codes_per_cell,
                    )
                else:
                    u_boot_j = u_j
                    map_boot_j = _map_for_target(
                        u_j.size,
                        group_id_to_psu_code,
                        eligible_group_ids,
                    )
                se_j, ci_j, p_j, _ = _bootstrap_one_target(
                    u_centered=u_boot_j,
                    divisor=n_j,
                    original=eff_j,
                    n_bootstrap=self.n_bootstrap,
                    weight_type=self.bootstrap_weights,
                    alpha=self.alpha,
                    rng=rng,
                    context="dCDH joiners DID_+ bootstrap",
                    return_distribution=False,
                    group_to_psu_map=map_boot_j,
                )
                results.joiners_se = se_j
                results.joiners_ci = ci_j
                results.joiners_p_value = p_j

        # --- Leavers (DID_-) ---
        if leavers_inputs is not None:
            u_l, n_l, eff_l, u_pp_l = leavers_inputs
            if u_l.size > 0 and n_l > 0:
                if psu_varies:
                    if u_pp_l is None:
                        raise ValueError(
                            "Cell-level bootstrap requires leavers' "
                            "per-cell IF tensor (U_centered_pp_leavers) "
                            "when PSU varies within group, got None."
                        )
                    u_boot_l, map_boot_l = _unroll_target_to_cells(
                        u_pp_l,
                        psu_codes_per_cell,
                    )
                else:
                    u_boot_l = u_l
                    map_boot_l = _map_for_target(
                        u_l.size,
                        group_id_to_psu_code,
                        eligible_group_ids,
                    )
                se_l, ci_l, p_l, _ = _bootstrap_one_target(
                    u_centered=u_boot_l,
                    divisor=n_l,
                    original=eff_l,
                    n_bootstrap=self.n_bootstrap,
                    weight_type=self.bootstrap_weights,
                    alpha=self.alpha,
                    rng=rng,
                    context="dCDH leavers DID_- bootstrap",
                    return_distribution=False,
                    group_to_psu_map=map_boot_l,
                )
                results.leavers_se = se_l
                results.leavers_ci = ci_l
                results.leavers_p_value = p_l

        # --- Placebo (DID_M^pl) ---
        # Phase 1 placebo has no per-cell IF (unpack tolerates None in
        # the fourth slot — callers always pass None for this target).
        if placebo_inputs is not None:
            u_pl, n_pl, eff_pl, _u_pp_pl_unused = placebo_inputs
            if u_pl.size > 0 and n_pl > 0:
                se_pl, ci_pl, p_pl, _ = _bootstrap_one_target(
                    u_centered=u_pl,
                    divisor=n_pl,
                    original=eff_pl,
                    n_bootstrap=self.n_bootstrap,
                    weight_type=self.bootstrap_weights,
                    alpha=self.alpha,
                    rng=rng,
                    context="dCDH placebo DID_M^pl bootstrap",
                    return_distribution=False,
                    group_to_psu_map=_map_for_target(
                        u_pl.size,
                        group_id_to_psu_code,
                        eligible_group_ids,
                    ),
                )
                results.placebo_se = se_pl
                results.placebo_ci = ci_pl
                results.placebo_p_value = p_pl

        # --- Phase 2: Multi-horizon bootstrap with shared weight matrix ---
        # Generate ONE shared weight matrix so all horizons use the same
        # bootstrap draw, making the sup-t statistic a valid joint
        # multiplier-bootstrap band. Under PSU-within-group-constant the
        # shared draws live at group granularity (bit-identical to
        # pre-cell-level); under within-group-varying PSU the shared
        # draws live at PSU granularity and are broadcast per-horizon to
        # the horizon's cells via `psu_codes_per_cell`.
        if multi_horizon_inputs is not None:
            es_ses: Dict[int, float] = {}
            es_cis: Dict[int, Tuple[float, float]] = {}
            es_pvals: Dict[int, float] = {}
            es_dists: Dict[int, np.ndarray] = {}

            n_groups_mh = n_groups_for_overall
            if psu_varies:
                # Draw ONE shared (n_bootstrap, n_psu) PSU-level weight
                # matrix. Broadcast per-horizon via each horizon's
                # cell-to-PSU map inside the loop. PSU count derived
                # from the dense code domain of psu_codes_per_cell.
                assert psu_codes_per_cell is not None
                valid_psu_codes = psu_codes_per_cell[psu_codes_per_cell >= 0]
                n_psu_mh = int(valid_psu_codes.max()) + 1 if valid_psu_codes.size > 0 else 0
                shared_psu_weights: Optional[np.ndarray]
                if n_psu_mh > 0:
                    shared_psu_weights = _generate_bootstrap_weights_batch(
                        n_bootstrap=self.n_bootstrap,
                        n_units=n_psu_mh,
                        weight_type=self.bootstrap_weights,
                        rng=rng,
                    )
                else:
                    shared_psu_weights = None
                shared_weights = None  # not used on the cell path
            else:
                # Shared weight matrix sized for the group set. Under
                # PSU-within-group-constant (Hall-Mammen wild PSU),
                # weights are drawn once per PSU and broadcast to groups
                # so all groups in the same PSU share a multiplier
                # within a single bootstrap replicate — preserving the
                # sup-t joint distribution across horizons.
                shared_weights = _generate_psu_or_group_weights(
                    n_bootstrap=self.n_bootstrap,
                    n_groups_target=n_groups_mh,
                    weight_type=self.bootstrap_weights,
                    rng=rng,
                    group_to_psu_map=_map_for_target(
                        n_groups_mh,
                        group_id_to_psu_code,
                        eligible_group_ids,
                    ),
                )
                shared_psu_weights = None

            for l_h, (u_h, n_h, eff_h, u_pp_h) in sorted(multi_horizon_inputs.items()):
                if u_h.size > 0 and n_h > 0:
                    # Under the current contract every horizon's IF
                    # vector uses the variance-eligible group ordering
                    # from `eligible_group_ids`, so the shared weight
                    # matrix is already at the right shape. Assert
                    # this invariant so any future refactor that
                    # introduces horizon-specific masking fails loudly
                    # rather than silently misaligning PSU clusters via
                    # positional truncation.
                    if u_h.size != n_groups_mh:
                        raise ValueError(
                            f"Multi-horizon bootstrap: horizon {l_h} "
                            f"IF vector has {u_h.size} entries but "
                            f"shared weight matrix has {n_groups_mh} "
                            f"columns. dCDH's contract requires every "
                            f"horizon to use the variance-eligible "
                            f"group ordering; to support a horizon "
                            f"with a different ordering, thread "
                            f"target-specific group IDs through "
                            f"`multi_horizon_inputs` and project the "
                            f"shared PSU draws onto the horizon's own "
                            f"ordering via `_map_for_target`."
                        )
                    if psu_varies:
                        if u_pp_h is None:
                            raise ValueError(
                                f"Cell-level bootstrap requires "
                                f"per-cell IF tensor for multi-horizon "
                                f"target l={l_h} when PSU varies "
                                f"within group, got None."
                            )
                        assert shared_psu_weights is not None
                        # Cell-level: unroll this horizon's cells and
                        # broadcast the shared PSU weights.
                        u_cell_h, psu_cell_h = _unroll_target_to_cells(
                            u_pp_h,
                            psu_codes_per_cell,
                        )
                        if u_cell_h.size == 0:
                            continue
                        w_cell_h = shared_psu_weights[:, psu_cell_h]
                        deviations = (w_cell_h @ u_cell_h) / n_h
                    else:
                        assert shared_weights is not None
                        deviations = (shared_weights @ u_h) / n_h
                    dist_h = deviations + eff_h

                    se_h, ci_h, p_h = _compute_effect_bootstrap_stats(
                        original_effect=eff_h,
                        boot_dist=dist_h,
                        alpha=self.alpha,
                    )
                    es_ses[l_h] = se_h
                    es_cis[l_h] = ci_h
                    es_pvals[l_h] = p_h
                    es_dists[l_h] = dist_h

            results.event_study_ses = es_ses
            results.event_study_cis = es_cis
            results.event_study_p_values = es_pvals

            # Sup-t simultaneous confidence bands using the shared draws.
            valid_horizons = [
                l_h
                for l_h in es_dists
                if l_h in es_ses and np.isfinite(es_ses[l_h]) and es_ses[l_h] > 0
            ]
            if len(valid_horizons) >= 2:
                boot_matrix = np.array([es_dists[l_h] for l_h in valid_horizons])
                effects_vec = np.array([multi_horizon_inputs[l_h][2] for l_h in valid_horizons])
                ses_vec = np.array([es_ses[l_h] for l_h in valid_horizons])
                t_stats = np.abs((boot_matrix - effects_vec[:, None]) / ses_vec[:, None])
                sup_t_dist = np.max(t_stats, axis=0)
                finite_mask = np.isfinite(sup_t_dist)
                if finite_mask.sum() > 0.5 * self.n_bootstrap:
                    cband_crit = float(np.quantile(sup_t_dist[finite_mask], 1 - self.alpha))
                    results.cband_crit_value = cband_crit

        # --- Phase 2: Placebo horizon bootstrap ---
        # Note: placebo-horizons are treated as independent single-target
        # draws (no sup-t joint-distribution requirement across placebo
        # horizons), so each horizon gets its own RNG draw via
        # `_bootstrap_one_target`. Under within-group-varying PSU the
        # per-horizon cell unroll is used.
        if placebo_horizon_inputs is not None:
            pl_ses: Dict[int, float] = {}
            pl_cis: Dict[int, Tuple[float, float]] = {}
            pl_pvals: Dict[int, float] = {}

            for l_h, (u_h, n_h, eff_h, u_pp_h) in sorted(placebo_horizon_inputs.items()):
                if u_h.size > 0 and n_h > 0:
                    if psu_varies:
                        if u_pp_h is None:
                            raise ValueError(
                                f"Cell-level bootstrap requires "
                                f"per-cell IF tensor for placebo-"
                                f"horizon target l={l_h} when PSU "
                                f"varies within group, got None."
                            )
                        u_boot_plh, map_boot_plh = _unroll_target_to_cells(
                            u_pp_h,
                            psu_codes_per_cell,
                        )
                        if u_boot_plh.size == 0:
                            continue
                    else:
                        u_boot_plh = u_h
                        map_boot_plh = _map_for_target(
                            u_h.size,
                            group_id_to_psu_code,
                            eligible_group_ids,
                        )
                    se_h, ci_h, p_h, _ = _bootstrap_one_target(
                        u_centered=u_boot_plh,
                        divisor=n_h,
                        original=eff_h,
                        n_bootstrap=self.n_bootstrap,
                        weight_type=self.bootstrap_weights,
                        alpha=self.alpha,
                        rng=rng,
                        context=f"dCDH placebo l={l_h} bootstrap",
                        return_distribution=False,
                        group_to_psu_map=map_boot_plh,
                    )
                    pl_ses[l_h] = se_h
                    pl_cis[l_h] = ci_h
                    pl_pvals[l_h] = p_h

            results.placebo_horizon_ses = pl_ses
            results.placebo_horizon_cis = pl_cis
            results.placebo_horizon_p_values = pl_pvals

        # --- Phase 3: Per-path (by_path) bootstrap ---
        # Treated as independent single-target draws per (path, horizon).
        # No shared-weight / sup-t joint distribution across paths (or
        # across horizons within a path) in this release — that's
        # Wave 2 item 4 follow-up work. Each target follows the same
        # `_bootstrap_one_target` dispatch as the single-horizon paths
        # above; the survey cell-level dispatch path is not reachable
        # here because the `by_path + survey_design` combination is
        # gated out in fit() before bootstrap is invoked.
        if path_bootstrap_inputs is not None:
            path_ses: Dict[Tuple[int, ...], Dict[int, float]] = {}
            path_cis: Dict[Tuple[int, ...], Dict[int, Tuple[float, float]]] = {}
            path_pvals: Dict[Tuple[int, ...], Dict[int, float]] = {}

            for path_key, horizon_inputs in path_bootstrap_inputs.items():
                path_ses[path_key] = {}
                path_cis[path_key] = {}
                path_pvals[path_key] = {}
                for l_h, (u_h, n_h, eff_h, _u_pp_h) in sorted(horizon_inputs.items()):
                    if u_h.size > 0 and n_h > 0:
                        map_boot_ph = _map_for_target(
                            u_h.size,
                            group_id_to_psu_code,
                            eligible_group_ids,
                        )
                        # np.errstate wrap: an identically-zero
                        # centered IF (degenerate path + horizon) would
                        # otherwise emit a RuntimeWarning: invalid
                        # value on the divide step in
                        # bootstrap_utils. The analytical pass has
                        # already emitted a UserWarning for the same
                        # (path, horizon); suppressing the
                        # RuntimeWarning here avoids stacked-warning
                        # noise without masking genuine numerical
                        # issues (`_bootstrap_one_target` still
                        # returns NaN SE for the zero-IF branch via
                        # the existing `se <= 0` guard in
                        # bootstrap_utils).
                        with np.errstate(invalid="ignore", divide="ignore"):
                            se_h, ci_h, p_h, _ = _bootstrap_one_target(
                                u_centered=u_h,
                                divisor=n_h,
                                original=eff_h,
                                n_bootstrap=self.n_bootstrap,
                                weight_type=self.bootstrap_weights,
                                alpha=self.alpha,
                                rng=rng,
                                context=f"dCDH by_path path={path_key} l={l_h} bootstrap",
                                return_distribution=False,
                                group_to_psu_map=map_boot_ph,
                            )
                        path_ses[path_key][l_h] = se_h
                        path_cis[path_key][l_h] = ci_h
                        path_pvals[path_key][l_h] = p_h

            results.path_ses = path_ses
            results.path_cis = path_cis
            results.path_p_values = path_pvals

        # --- Phase 3: Per-path placebo (by_path + placebo) bootstrap ---
        # Sibling of the per-path event-study block above for backward
        # placebo lags. Same independent single-target dispatch per
        # (path, lag_l) via `_bootstrap_one_target`; the survey cell-
        # level path is unreachable here because
        # `by_path + survey_design` is gated out in fit() before
        # bootstrap is invoked. The `np.errstate` wrap mirrors the
        # event-study block's degenerate-IF stacked-warning suppression.
        if path_placebo_bootstrap_inputs is not None:
            path_pl_ses: Dict[Tuple[int, ...], Dict[int, float]] = {}
            path_pl_cis: Dict[Tuple[int, ...], Dict[int, Tuple[float, float]]] = {}
            path_pl_pvals: Dict[Tuple[int, ...], Dict[int, float]] = {}

            for path_key, horizon_inputs in path_placebo_bootstrap_inputs.items():
                path_pl_ses[path_key] = {}
                path_pl_cis[path_key] = {}
                path_pl_pvals[path_key] = {}
                for lag_l, (u_pl, n_pl, eff_pl, _u_pp_pl) in sorted(horizon_inputs.items()):
                    if u_pl.size > 0 and n_pl > 0:
                        map_boot_pl_path = _map_for_target(
                            u_pl.size,
                            group_id_to_psu_code,
                            eligible_group_ids,
                        )
                        with np.errstate(invalid="ignore", divide="ignore"):
                            (
                                se_pl_h,
                                ci_pl_h,
                                p_pl_h,
                                _,
                            ) = _bootstrap_one_target(
                                u_centered=u_pl,
                                divisor=n_pl,
                                original=eff_pl,
                                n_bootstrap=self.n_bootstrap,
                                weight_type=self.bootstrap_weights,
                                alpha=self.alpha,
                                rng=rng,
                                context=(
                                    f"dCDH by_path placebo " f"path={path_key} l={lag_l} bootstrap"
                                ),
                                return_distribution=False,
                                group_to_psu_map=map_boot_pl_path,
                            )
                        path_pl_ses[path_key][lag_l] = se_pl_h
                        path_pl_cis[path_key][lag_l] = ci_pl_h
                        path_pl_pvals[path_key][lag_l] = p_pl_h

            results.path_placebo_ses = path_pl_ses
            results.path_placebo_cis = path_pl_cis
            results.path_placebo_p_values = path_pl_pvals

        # --- Phase 3: Per-path joint sup-t (by_path + n_bootstrap > 0) ---
        # Sibling of the OVERALL event-study sup-t at the multi-horizon
        # block above (`:599-614`). Per-path joint simultaneous
        # confidence bands across horizons 1..L_max within each path:
        # one shared (n_bootstrap, n_eligible) multiplier weight matrix
        # (using `self.bootstrap_weights` — Rademacher / Mammen / Webb)
        # per path is broadcast across all valid horizons of that path,
        # producing correlated bootstrap distributions across horizons.
        # The path-specific critical value
        # `c_p = quantile(max_l |t_l|, 1-alpha)` is the band half-width
        # multiplier applied to each horizon's bootstrap SE in fit().
        #
        # Note (asymmetry vs OVERALL): this draws a FRESH shared-weights
        # matrix per path AFTER the per-path SE block above has populated
        # results.path_ses via independent per-(path, horizon) draws.
        # Numerator: fresh shared draws; denominator: bootstrap SEs from
        # the earlier independent draws. Asymptotically equivalent to
        # OVERALL's self-consistent reuse, but NOT bit-identical. The
        # fresh draw is intentional: it preserves RNG-state isolation
        # for existing per-path SE seed-reproducibility tests.
        #
        # Gates: a path needs >=2 valid horizons (finite bootstrap SE>0)
        # AND a strict majority (>50%) of finite sup-t draws to receive
        # a band. Otherwise the path is absent from
        # path_cband_crit_values (mirrors OVERALL absent-key pattern at
        # `:605,612`; the strict-majority gate matches the OVERALL
        # `finite_mask.sum() > 0.5 * n_bootstrap` semantics — exactly
        # half finite is NOT enough).
        if path_bootstrap_inputs is not None and results.path_ses:
            path_cband_crits: Dict[Tuple[int, ...], float] = {}
            path_cband_n_valid: Dict[Tuple[int, ...], int] = {}

            for path_key, horizon_inputs in path_bootstrap_inputs.items():
                bs_ses_for_path = results.path_ses.get(path_key, {})
                valid_horizon_stats = []
                for l_h, (u_h, n_h, eff_h, _u_pp_h) in sorted(horizon_inputs.items()):
                    if u_h.size == 0 or n_h <= 0:
                        continue
                    bs_se = bs_ses_for_path.get(l_h, np.nan)
                    if not np.isfinite(bs_se) or bs_se <= 0:
                        continue
                    valid_horizon_stats.append((l_h, u_h, n_h, eff_h, bs_se))

                if len(valid_horizon_stats) < 2:
                    continue

                # All horizons within a path use the same n_eligible
                # (variance-eligible group ordering enforced by
                # _collect_path_bootstrap_inputs's use of
                # eligible_mask_var for cohort-recentering); use the
                # first valid horizon's IF size as the shared dim.
                n_dim = valid_horizon_stats[0][1].size
                map_path = _map_for_target(
                    n_dim,
                    group_id_to_psu_code,
                    eligible_group_ids,
                )
                with np.errstate(invalid="ignore", divide="ignore"):
                    shared_weights = _generate_psu_or_group_weights(
                        n_bootstrap=self.n_bootstrap,
                        n_groups_target=n_dim,
                        weight_type=self.bootstrap_weights,
                        rng=rng,
                        group_to_psu_map=map_path,
                    )
                    es_dists_path = []
                    for _l_h, u_h, n_h, eff_h, _bs_se in valid_horizon_stats:
                        deviations = (shared_weights @ u_h) / n_h
                        es_dists_path.append(eff_h + deviations)
                    boot_matrix = np.asarray(es_dists_path)
                    effects_vec = np.array([v[3] for v in valid_horizon_stats])
                    ses_vec = np.array([v[4] for v in valid_horizon_stats])
                    t_stats = np.abs((boot_matrix - effects_vec[:, None]) / ses_vec[:, None])
                    sup_t_dist = np.max(t_stats, axis=0)
                    finite_mask = np.isfinite(sup_t_dist)
                    if finite_mask.sum() <= 0.5 * self.n_bootstrap:
                        continue
                    crit_p = float(np.quantile(sup_t_dist[finite_mask], 1.0 - self.alpha))

                if not np.isfinite(crit_p):
                    continue

                path_cband_crits[path_key] = crit_p
                path_cband_n_valid[path_key] = len(valid_horizon_stats)

            results.path_cband_crit_values = path_cband_crits
            results.path_cband_n_valid_horizons = path_cband_n_valid

        return results


# =============================================================================
# Internal helpers
# =============================================================================


def _psu_varies_within_group(
    psu_codes_per_cell: Optional[np.ndarray],
) -> bool:
    """True when any row of ``psu_codes_per_cell`` has more than one
    unique PSU label (ignoring -1 sentinel entries).

    When ``False`` — including the ``None`` case for non-survey fits —
    the legacy group-level bootstrap path is invoked. The row-sum
    identity ``sum_{c in g} u_cell[c] == u_centered[g]`` established
    by ``_cohort_recenter_per_period`` makes the cell-level and
    group-level bootstraps statistically equivalent under this regime,
    and the group-level path is bit-identical to pre-cell-level
    releases through the existing identity-map fast path.
    """
    if psu_codes_per_cell is None:
        return False
    for row in psu_codes_per_cell:
        valid = row[row >= 0]
        if valid.size > 1 and np.unique(valid).size > 1:
            return True
    return False


def _unroll_target_to_cells(
    u_per_period_target: np.ndarray,
    psu_codes_per_cell: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Flatten a target's cohort-recentered per-cell IF tensor + its
    per-cell PSU map into 1-D arrays, dropping cells with sentinel
    PSU code (-1 — zero-weight cells).

    Both inputs must have shape ``(n_eligible_groups, n_periods)``;
    the dCDH bootstrap contract guarantees all targets share the
    variance-eligible group ordering, so no per-target row subset
    is needed.

    Raises ``ValueError`` when any sentinel cell (-1 PSU) carries
    non-zero cohort-recentered IF mass. This is a supported-edge-
    case guard: under terminal missingness, ``_cohort_recenter_per_period``
    subtracts column means across the full period grid, so a group
    with no observation at period ``t`` can acquire non-zero centered
    mass at that sentinel cell. The cell-level bootstrap cannot
    allocate that mass to any PSU (the cell has no positive-weight
    obs), so silently dropping it would under-weight the group's
    bootstrap contribution. The analytical TSL path shares the same
    cell-period allocator and fires a matching guard in
    ``_survey_se_from_group_if``, so both paths reject this regime
    consistently. **Documented workaround (bootstrap path):**
    pre-process the panel to remove terminal missingness (drop
    late-exit groups or trim to a balanced sub-panel). The within-
    group-varying-PSU bootstrap has no allocator fallback — unlike
    Binder TSL, where using an explicit ``psu=<group_col>`` routes
    the analytical path through the legacy group-level allocator;
    that fallback is not available on the bootstrap side because
    the cell-level wild PSU bootstrap IS the varying-PSU regime.

    Returns ``(u_cell, psu_cell)`` of shape
    ``(n_valid_cells_in_target,)`` each.
    """
    if psu_codes_per_cell is None:
        raise ValueError(
            "_unroll_target_to_cells requires psu_codes_per_cell; "
            "caller should only invoke this on the cell-level path."
        )
    if u_per_period_target.shape != psu_codes_per_cell.shape:
        raise ValueError(
            "Cell-level bootstrap shape mismatch: target per-period "
            f"IF tensor has shape {u_per_period_target.shape} but "
            f"psu_codes_per_cell has shape {psu_codes_per_cell.shape}. "
            "The dCDH bootstrap contract requires every target's "
            "per-cell tensor to align with the (n_eligible_groups, "
            "n_periods) layout of psu_codes_per_cell."
        )
    flat_u = u_per_period_target.ravel()
    flat_psu = psu_codes_per_cell.ravel()
    mask = flat_psu >= 0
    # Sentinel-mass guard: reject terminal-missingness + within-group-
    # varying PSU + bootstrap. The cohort-recentering column-subtraction
    # at `_cohort_recenter_per_period` can leak non-zero centered mass
    # onto cells with no positive-weight obs (missing-cell rows in the
    # cohort still get -col_mean added when other rows contribute at
    # that column). Dropping that mass silently would under-cluster the
    # bootstrap in a supported panel regime.
    sentinel_mass = flat_u[~mask]
    if sentinel_mass.size > 0 and bool(np.any(np.abs(sentinel_mass) > 1e-12)):
        raise ValueError(
            "Cell-level bootstrap cannot be computed on this survey "
            "panel: cohort-recentered IF mass landed on cells with "
            "no positive-weight observations (psu_codes_per_cell == "
            "-1). This typically occurs when terminal missingness "
            "(groups observed only through some period) combines with "
            "within-group-varying PSU: `_cohort_recenter_per_period` "
            "subtracts column means across the full period grid, so a "
            "group with no observation at period t acquires non-zero "
            "centered mass there, which the cell-level allocator "
            "cannot allocate to any PSU. The analytical TSL path "
            "(`_survey_se_from_group_if`) fires a matching guard on "
            "the same regime, so both paths reject this panel "
            "consistently. Pre-process the panel to remove terminal "
            "missingness (drop late-exit groups or trim to a balanced "
            "sub-panel). The within-group-varying-PSU bootstrap has "
            "no allocator fallback — unlike Binder TSL, switching to "
            "`psu=<group_col>` does not help here because the varying-"
            "PSU bootstrap IS the cell-level path, not an analytical "
            "surface with a legacy-allocator alternative."
        )
    return flat_u[mask], flat_psu[mask].astype(np.int64, copy=False)


def _map_for_target(
    target_size: int,
    group_id_to_psu_code: Optional[Dict[Any, int]],
    eligible_group_ids: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """Build a PSU map for a bootstrap target from IDs (not positions).

    The caller passes:
    - ``group_id_to_psu_code``: a dict mapping each variance-eligible
      group ID to its dense PSU code (built once in ``fit()``).
    - ``eligible_group_ids``: the ordered list of group IDs that
      correspond to the current target's ``u_centered`` vector.

    Returns an integer array of length ``target_size`` where entry
    ``i`` is the PSU code for the ``i``-th contributing group.

    Returns ``None`` when no PSU information is available (plain
    multiplier-bootstrap path — identity across targets).

    **Group-level granularity only.** This helper is invoked on the
    legacy group-level bootstrap path; on the cell-level path
    (within-group-varying PSU), callers build the cell-to-PSU map
    directly via ``_unroll_target_to_cells`` and do not route through
    ``_map_for_target``, so the size-mismatch raise below does not
    fire.

    Raises ``ValueError`` if ``target_size`` does not match
    ``len(eligible_group_ids)``: every current dCDH bootstrap target
    uses the variance-eligible group ordering on the group-level path,
    so any size mismatch signals that a caller introduced a target
    whose group subset diverges and should pass its own
    ``target_group_ids`` rather than reusing the overall eligible
    list. Also raises ``ValueError`` if any group ID is missing from
    the dict (signaling misalignment between the target's IF vector
    and the map's keys).
    """
    if group_id_to_psu_code is None or eligible_group_ids is None:
        return None
    if target_size != len(eligible_group_ids):
        raise ValueError(
            f"Bootstrap target size ({target_size}) does not match "
            f"eligible_group_ids length ({len(eligible_group_ids)}). "
            "dCDH's bootstrap contract requires all current targets to "
            "use the variance-eligible group ordering; if a new target "
            "has a different ordering, pass target-specific group IDs "
            "to _map_for_target rather than reusing eligible_group_ids."
        )
    try:
        return np.array(
            [group_id_to_psu_code[gid] for gid in eligible_group_ids],
            dtype=np.int64,
        )
    except KeyError as e:
        raise ValueError(
            f"Group ID {e.args[0]!r} in eligible_group_ids has no entry "
            f"in group_id_to_psu_code — PSU map is misaligned with the "
            f"bootstrap target's group set."
        ) from e


def _generate_psu_or_group_weights(
    n_bootstrap: int,
    n_groups_target: int,
    weight_type: str,
    rng: np.random.Generator,
    group_to_psu_map: Optional[np.ndarray],
) -> np.ndarray:
    """Generate a group-level weight matrix, possibly via PSU broadcasting.

    When ``group_to_psu_map`` is ``None`` or is the identity (each group
    is its own PSU), generates weights at the group level directly —
    bit-identical to the pre-PSU-bootstrap contract.

    When ``group_to_psu_map`` has fewer unique values than
    ``n_groups_target`` (strictly coarser PSU than group), generates
    weights at the PSU level and broadcasts to groups via the map.
    This is the Hall-Mammen wild PSU bootstrap.

    Parameters
    ----------
    n_bootstrap, weight_type, rng
        Passed through to generate_bootstrap_weights_batch.
    n_groups_target : int
        Number of groups contributing to the target's IF vector.
    group_to_psu_map : np.ndarray or None
        Dense integer PSU indices of shape ``(n_groups_target,)``.
        ``None`` triggers the group-level path.

    Returns
    -------
    np.ndarray
        Shape ``(n_bootstrap, n_groups_target)`` multiplier weights.
    """
    if group_to_psu_map is None:
        return _generate_bootstrap_weights_batch(
            n_bootstrap=n_bootstrap,
            n_units=n_groups_target,
            weight_type=weight_type,
            rng=rng,
        )
    if len(group_to_psu_map) != n_groups_target:
        raise ValueError(
            f"group_to_psu_map length ({len(group_to_psu_map)}) does not "
            f"match n_groups_target ({n_groups_target})."
        )
    n_psu = int(np.max(group_to_psu_map)) + 1 if group_to_psu_map.size > 0 else 0
    if n_psu >= n_groups_target:
        # Identity (each group its own PSU) — skip the broadcast for a
        # bit-identical fast path matching the pre-PSU-bootstrap behavior.
        return _generate_bootstrap_weights_batch(
            n_bootstrap=n_bootstrap,
            n_units=n_groups_target,
            weight_type=weight_type,
            rng=rng,
        )
    # Hall-Mammen wild PSU bootstrap: draw n_psu multipliers, broadcast
    # via the dense index map so all groups in the same PSU share a
    # multiplier. Preserves clustered sampling structure.
    psu_weights = _generate_bootstrap_weights_batch(
        n_bootstrap=n_bootstrap,
        n_units=n_psu,
        weight_type=weight_type,
        rng=rng,
    )
    return psu_weights[:, group_to_psu_map]


def _bootstrap_one_target(
    u_centered: np.ndarray,
    divisor: int,
    original: float,
    n_bootstrap: int,
    weight_type: str,
    alpha: float,
    rng: np.random.Generator,
    context: str,
    return_distribution: bool,
    group_to_psu_map: Optional[np.ndarray] = None,
) -> Tuple[float, Tuple[float, float], float, Optional[np.ndarray]]:
    """
    Run the multiplier bootstrap for a single dCDH target.

    Generates an ``(n_bootstrap, len(u_centered))`` matrix of multiplier
    weights, multiplies by ``u_centered``, and divides by ``divisor`` to
    get a bootstrap distribution. Returns
    ``(se, (ci_lo, ci_hi), p_value, distribution)``; ``distribution`` is
    ``None`` when ``return_distribution=False`` (saves memory for
    auxiliary targets).

    The "centered" naming is important: this function expects
    ``u_centered`` to already have its cohort means subtracted (so the
    sample mean of the bootstrap distribution should be approximately
    zero, not the original effect). The original effect is passed
    separately as the centering point for the percentile p-value.

    When ``group_to_psu_map`` is provided (length ``len(u_centered)``,
    dense integer PSU indices), multiplier weights are generated at the
    PSU level and broadcast to groups so all groups in the same PSU
    receive the same bootstrap multiplier. This is the Hall-Mammen wild
    PSU bootstrap; it reduces to the group-level bootstrap when each
    group is its own PSU (identity map).
    """
    n_groups_target = u_centered.shape[0]
    if n_groups_target == 0 or divisor == 0:
        return np.nan, (np.nan, np.nan), np.nan, None

    weight_matrix = _generate_psu_or_group_weights(
        n_bootstrap=n_bootstrap,
        n_groups_target=n_groups_target,
        weight_type=weight_type,
        rng=rng,
        group_to_psu_map=group_to_psu_map,
    )

    # Each bootstrap replicate: (1 / divisor) * sum_g w_b[g] * u_centered[g]
    # The result is the bootstrap analog of the *deviation* from the original
    # effect; shift it by `original` so the bootstrap distribution is centered
    # at the original point estimate (which is what compute_effect_bootstrap_stats
    # expects when computing the percentile p-value).
    deviations = (weight_matrix @ u_centered) / divisor
    boot_dist = original + deviations

    se, ci, p_value = _compute_effect_bootstrap_stats(
        original_effect=original,
        boot_dist=boot_dist,
        alpha=alpha,
        context=context,
    )

    return se, ci, p_value, (boot_dist if return_distribution else None)
