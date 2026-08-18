"""
Bootstrap inference for Callaway-Sant'Anna estimator.

This module provides the bootstrap results container and the mixin class
with bootstrap inference methods. Weight generation and statistical helpers
are in :mod:`diff_diff.bootstrap_utils`.
"""

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Iterator, List, Optional, Tuple

import numpy as np

from diff_diff.bootstrap_chunking import (
    ReplayableWeightStream,
    compute_block_size,
    effective_weight_backend,
    iter_survey_multiplier_weight_blocks,
    iter_weight_blocks,
    tiled_if_matmul,
)
from diff_diff.bootstrap_utils import (
    compute_bootstrap_pvalue as _compute_bootstrap_pvalue_func,
)
from diff_diff.bootstrap_utils import (
    compute_effect_bootstrap_stats as _compute_effect_bootstrap_stats_func,
)
from diff_diff.bootstrap_utils import (
    compute_effect_bootstrap_stats_batch as _compute_effect_bootstrap_stats_batch_func,
)
from diff_diff.bootstrap_utils import (
    compute_percentile_ci as _compute_percentile_ci_func,
)
from diff_diff.utils import safe_inference_batch

if TYPE_CHECKING:
    import pandas as pd

    from diff_diff.staggered_aggregation import PrecomputedData


# =============================================================================
# Bootstrap Results Container
# =============================================================================


@dataclass
class CSBootstrapResults:
    """
    Results from Callaway-Sant'Anna multiplier bootstrap inference.

    Attributes
    ----------
    n_bootstrap : int
        Number of bootstrap iterations.
    weight_type : str
        Type of bootstrap weights used.
    alpha : float
        Significance level used for confidence intervals.
    overall_att_se : float
        Bootstrap standard error for overall ATT.
    overall_att_ci : Tuple[float, float]
        Bootstrap confidence interval for overall ATT.
    overall_att_p_value : float
        Bootstrap p-value for overall ATT.
    group_time_ses : Dict[Tuple[Any, Any], float]
        Bootstrap SEs for each ATT(g,t).
    group_time_cis : Dict[Tuple[Any, Any], Tuple[float, float]]
        Bootstrap CIs for each ATT(g,t).
    group_time_p_values : Dict[Tuple[Any, Any], float]
        Bootstrap p-values for each ATT(g,t).
    event_study_ses : Optional[Dict[int, float]]
        Bootstrap SEs for event study effects.
    event_study_cis : Optional[Dict[int, Tuple[float, float]]]
        Bootstrap CIs for event study effects.
    event_study_p_values : Optional[Dict[int, float]]
        Bootstrap p-values for event study effects.
    group_effect_ses : Optional[Dict[Any, float]]
        Bootstrap SEs for group effects.
    group_effect_cis : Optional[Dict[Any, Tuple[float, float]]]
        Bootstrap CIs for group effects.
    group_effect_p_values : Optional[Dict[Any, float]]
        Bootstrap p-values for group effects.
    bootstrap_distribution : Optional[np.ndarray]
        Full bootstrap distribution of overall ATT (if requested).
    overall_att_es_se : Optional[float]
        Bootstrap standard error for the paper Eq. 4.14 overall (event-study average).
    overall_att_es_ci : Optional[Tuple[float, float]]
        Bootstrap confidence interval for the Eq. 4.14 overall.
    overall_att_es_p_value : Optional[float]
        Bootstrap p-value for the Eq. 4.14 overall.
    """

    n_bootstrap: int
    weight_type: str
    alpha: float
    overall_att_se: float
    overall_att_ci: Tuple[float, float]
    overall_att_p_value: float
    group_time_ses: Dict[Tuple[Any, Any], float]
    group_time_cis: Dict[Tuple[Any, Any], Tuple[float, float]]
    group_time_p_values: Dict[Tuple[Any, Any], float]
    event_study_ses: Optional[Dict[int, float]] = None
    event_study_cis: Optional[Dict[int, Tuple[float, float]]] = None
    event_study_p_values: Optional[Dict[int, float]] = None
    group_effect_ses: Optional[Dict[Any, float]] = None
    group_effect_cis: Optional[Dict[Any, Tuple[float, float]]] = None
    group_effect_p_values: Optional[Dict[Any, float]] = None
    bootstrap_distribution: Optional[np.ndarray] = field(default=None, repr=False)
    cband_crit_value: Optional[float] = None
    # Paper Eq. (4.14) overall (event-study average) bootstrap inference.
    overall_att_es_se: Optional[float] = None
    overall_att_es_ci: Optional[Tuple[float, float]] = None
    overall_att_es_p_value: Optional[float] = None

    def __post_init__(self) -> None:
        # Post-fit replay bookkeeping, attached as PLAIN attributes (never
        # dataclass fields) so the exported class's __init__ signature,
        # dataclasses.fields() and asdict() stay unchanged — the same
        # private-carrier pattern the results objects use for retained
        # state. `_replay_bitgen_state` is the RNG snapshot taken at
        # weight-stream construction; `_replay_backend` is the
        # generation-branch identity ("rust"/"numpy", or "portable" for
        # provably backend-independent branches). Both are populated by
        # _run_multiplier_bootstrap on every run and pickle via __dict__.
        self._replay_bitgen_state: Optional[Dict[str, Any]] = None
        self._replay_backend: Optional[str] = None


# =============================================================================
# Bootstrap Mixin Class
# =============================================================================


class CallawaySantAnnaBootstrapMixin:
    """
    Mixin class providing bootstrap inference methods for CallawaySantAnna.

    This class is not intended to be used standalone. It provides methods
    that are used by the main CallawaySantAnna class for multiplier bootstrap
    inference.
    """

    # Type hints for attributes accessed from the main class
    # Host-supplied estimator name for user-facing bootstrap warnings. Declared
    # here because mypy type-checks the mixin independently of its hosts, so
    # `self._BOOTSTRAP_LABEL` would otherwise be [attr-defined]. ClassVar, not a
    # bare annotation: the hosts set it as class-level constant data, and an
    # instance-variable declaration here would make each of those a
    # "cannot override instance variable with class variable" [misc] error.
    # (Contrast `_warn_frame_offset`, which IS assigned via instance and so must
    # NOT be a ClassVar.)
    _BOOTSTRAP_LABEL: ClassVar[str]
    n_bootstrap: int
    bootstrap_weights: str
    alpha: float
    seed: Optional[int]
    anticipation: int

    if TYPE_CHECKING:

        def _compute_combined_influence_function(
            self,
            gt_pairs: List[Tuple[Any, Any]],
            weights: np.ndarray,
            effects: np.ndarray,
            groups_for_gt: np.ndarray,
            influence_func_info: Dict,
            # Optional to match the aggregation mixin, whose implementation this
            # stub shadows: post-fit aggregation runs from the retained kit with
            # no frame. A narrower stub here makes the two base classes
            # incompatible wherever both are mixed in.
            df: Optional["pd.DataFrame"],
            unit: Optional[str],
            precomputed: Optional["PrecomputedData"] = None,
            global_unit_to_idx: Optional[Dict[Any, int]] = None,
            n_global_units: Optional[int] = None,
        ) -> Tuple[np.ndarray, Optional[List]]: ...

    def _run_multiplier_bootstrap(
        self,
        group_time_effects: Dict[Tuple[Any, Any], Dict[str, Any]],
        influence_func_info: Dict[Tuple[Any, Any], Dict[str, Any]],
        aggregate: Optional[str],
        balance_e: Optional[int],
        treatment_groups: List[Any],
        time_periods: List[Any],
        df: Any = None,
        unit: Optional[str] = None,
        precomputed: Any = None,
        cband: bool = True,
        *,
        _replay_bitgen_state: Optional[Dict[str, Any]] = None,
    ) -> CSBootstrapResults:
        """
        Run multiplier bootstrap for inference on all parameters.

        ``_replay_bitgen_state`` (keyword-only, package-internal): a
        bit-generator state captured by a previous run. When given, the RNG
        is restored to it instead of seeding from ``self.seed``, so the
        multiplier-weight stream replays bit-identically — the post-fit
        ``aggregate()`` replay path. Valid only under the SAME weight
        backend that captured it (see
        :func:`diff_diff.bootstrap_chunking.effective_weight_backend`);
        the caller enforces the backend guard.

        This implements the multiplier bootstrap procedure from Callaway & Sant'Anna (2021).
        The key idea is to perturb the influence function contributions with random
        weights at the cluster (unit) level, then recompute aggregations.

        Parameters
        ----------
        group_time_effects : dict
            Dictionary of ATT(g,t) effects with analytical SEs.
        influence_func_info : dict
            Dictionary mapping (g,t) to influence function information.
        aggregate : str, optional
            Type of aggregation requested.
        balance_e : int, optional
            Balance parameter for event study.
        treatment_groups : list
            List of treatment cohorts.
        time_periods : list
            List of time periods.

        Returns
        -------
        CSBootstrapResults
            Bootstrap inference results.
        """
        # Warn about low bootstrap iterations. This site is USER-attributed, and
        # the DDD engine reaches it through one or two extra frames depending on
        # which surface was called, so it consults the offset the engine mirrors
        # onto the instance for the duration of a fit. CallawaySantAnna never
        # sets the attribute, so its attribution is bit-identical to 3.x.
        if self.n_bootstrap < 50:
            warnings.warn(
                f"n_bootstrap={self.n_bootstrap} is low. Consider n_bootstrap >= 199 "
                "for reliable inference. Percentile confidence intervals and p-values "
                "may be unreliable with few iterations.",
                UserWarning,
                stacklevel=3 + getattr(self, "_warn_frame_offset", 0),
            )

        rng = np.random.default_rng(self.seed)
        if _replay_bitgen_state is not None:
            # Post-fit replay: restore the exact state the fit-time run
            # captured at weight-stream construction. Nothing below consumes
            # the rng before that point, so the stream is bit-identical.
            rng.bit_generator.state = _replay_bitgen_state

        # Use global unit set for correct pg = n_g / N_total scaling.
        # Without this, pg is overestimated in unbalanced panels where some
        # units don't appear in any influence function.
        if precomputed is not None:
            all_units = precomputed["all_units"]
            n_units = precomputed.get("canonical_size", len(all_units))
            unit_to_idx = precomputed["unit_to_idx"]  # None for RCS
        else:
            # Fallback: collect units from influence functions. Needs the
            # per-cell unit-LABEL arrays, which in-package fits stopped
            # materializing (v3.8 per-cell allocation shave) — every
            # in-package caller threads `precomputed`, so this branch is
            # unreachable from a fit. Direct callers must thread
            # `precomputed` or supply label arrays.
            all_units_set = set()
            for (g, t), info in influence_func_info.items():
                t_units = info.get("treated_units")
                c_units = info.get("control_units")
                if t_units is None or c_units is None:
                    raise ValueError(
                        "Multiplier bootstrap without `precomputed` requires "
                        "per-cell 'treated_units'/'control_units' label arrays "
                        "in influence_func_info; in-package fits no longer "
                        "materialize them. Thread `precomputed` (as all "
                        "in-package callers do), or add the label arrays to "
                        "your influence_func_info."
                    )
                all_units_set.update(t_units)
                all_units_set.update(c_units)
            all_units = sorted(all_units_set)
            # Use global N from dataframe when available
            n_units = (
                df[unit].nunique() if (df is not None and unit is not None) else len(all_units)
            )
            unit_to_idx = {u: i for i, u in enumerate(all_units)}

        # Get list of (g,t) pairs that have influence function info
        # (skip zero-mass cells that recorded NaN ATT without IF)
        gt_pairs = [gt for gt in group_time_effects.keys() if gt in influence_func_info]
        n_gt = len(gt_pairs)

        # Identify post-treatment (g,t) pairs for overall ATT
        # Pre-treatment effects are for parallel trends assessment, not aggregated
        post_treatment_mask = np.array([t >= g - self.anticipation for (g, t) in gt_pairs])
        post_treatment_indices = np.where(post_treatment_mask)[0]

        # Compute aggregation weights for overall ATT (post-treatment only)
        # When survey weights are present, use fixed cohort survey masses
        # (from precomputed survey_weights × unit_cohorts), matching the
        # analytical _aggregate_simple() path in staggered_aggregation.py.
        # Do NOT use per-cell survey_weight_sum (which varies by cell on
        # unbalanced panels).
        # Fixed per-cohort aggregation masses — the SAME single source of truth
        # the analytical _aggregate_simple() / _aggregate_event_study() use, so
        # the bootstrap weights an unbalanced-panel (allow_unbalanced_panel) RC
        # aggregation by fixed UNIT cohort mass, not observation count. Returns
        # None for panel non-survey (→ per-cell agg_weight/n_treated fallback).
        from diff_diff.staggered_aggregation import fixed_cohort_agg_weights

        _fixed_masses = fixed_cohort_agg_weights(precomputed)

        def _agg_mass(gt):
            g = gt[0]
            if _fixed_masses is not None and g in _fixed_masses:
                return _fixed_masses[g]
            return group_time_effects[gt].get("agg_weight", group_time_effects[gt]["n_treated"])

        all_n_treated = np.array([_agg_mass(gt) for gt in gt_pairs], dtype=float)
        post_n_treated = all_n_treated[post_treatment_mask]

        # Filter out NaN ATT(g,t) cells from overall aggregation (matches analytical path)
        post_effects_raw = np.array(
            [group_time_effects[gt_pairs[i]]["effect"] for i in post_treatment_indices]
        )
        finite_post = np.isfinite(post_effects_raw)
        if not np.all(finite_post):
            post_treatment_indices = post_treatment_indices[finite_post]
            post_n_treated = post_n_treated[finite_post]

        # Flag to skip overall ATT aggregation when no post-treatment effects
        # But continue bootstrap for per-effect SEs (pre-treatment effects need bootstrap SEs too)
        skip_overall_aggregation = False
        if len(post_treatment_indices) == 0:
            warnings.warn(
                "No post-treatment effects for bootstrap aggregation. "
                "Overall ATT statistics will be NaN, but per-effect SEs will be computed.",
                UserWarning,
                stacklevel=2,
            )
            skip_overall_aggregation = True
            overall_weights_post = np.array([])
        else:
            overall_weights_post = post_n_treated / np.sum(post_n_treated)

        # Original point estimates
        original_atts = np.array([group_time_effects[gt]["effect"] for gt in gt_pairs])
        if skip_overall_aggregation:
            original_overall = np.nan
        else:
            original_overall = np.sum(overall_weights_post * original_atts[post_treatment_indices])

        # Prepare event study and group aggregation info if needed
        event_study_info = None
        group_agg_info = None

        if aggregate in ["event_study", "all"]:
            event_study_info = self._prepare_event_study_aggregation(
                gt_pairs,
                group_time_effects,
                balance_e,
                influence_func_info=influence_func_info,
                df=df,
                unit=unit,
                precomputed=precomputed,
                global_unit_to_idx=unit_to_idx,
                n_global_units=n_units,
            )

        if aggregate in ["group", "all"]:
            group_agg_info = self._prepare_group_aggregation(
                gt_pairs, group_time_effects, treatment_groups
            )

        # Pre-compute unit index arrays for each (g,t) pair (done once, not per iteration)
        gt_treated_indices = []
        gt_control_indices = []
        gt_treated_inf = []
        gt_control_inf = []

        for j, gt in enumerate(gt_pairs):
            info = influence_func_info[gt]
            gt_treated_indices.append(info["treated_idx"])
            gt_control_indices.append(info["control_idx"])
            gt_treated_inf.append(np.asarray(info["treated_inf"]))
            gt_control_inf.append(np.asarray(info["control_inf"]))

        # Generate bootstrap weights — PSU-level when survey design is present,
        # unit-level otherwise.
        resolved_survey_unit = (
            precomputed.get("resolved_survey_unit") if precomputed is not None else None
        )
        _use_survey_bootstrap = resolved_survey_unit is not None and (
            resolved_survey_unit.strata is not None
            or resolved_survey_unit.psu is not None
            or resolved_survey_unit.fpc is not None
        )

        # When the bootstrap routes through PSU-multiplier weights, the
        # bootstrap variance is unidentified if there are fewer than 2
        # PSUs (single-cluster designs collapse all multiplier draws to
        # constants → ≈0 variance from BLAS roundoff, NOT NaN). Without
        # this guard, downstream safe_inference would silently produce
        # tight CIs and near-zero p-values for a variance that's actually
        # undefined. Capture the flag here and NaN-out all bootstrap
        # inference surfaces before return (per feedback_no_silent_failures).
        _bootstrap_cluster_variance_unidentified = False

        if _use_survey_bootstrap:
            # The flag definition above guarantees this (mypy can't track it).
            assert resolved_survey_unit is not None
            # PSU-level multiplier weights, generated AND expanded one draw-block
            # at a time so the (n_bootstrap, n_units) matrix is never built in
            # full. This is the dominant allocation at large n_units, including
            # the default unit-level bootstrap (cluster=None, equivalently
            # cluster="unit": each unit its own PSU, n_psu == n_units).
            # Unstratified designs tile the generation; stratified designs (few
            # PSUs) fall back to full generation + sliced blocks.
            _block_size = compute_block_size(n_units, self.n_bootstrap)
            # Resolve psu_ids WITHOUT calling the generator: the stratified
            # branch of iter_survey_multiplier_weight_blocks draws from the rng
            # eagerly at call time, and the replayable stream below must
            # snapshot the rng state before any draw. This duplicates the
            # rng-free resolution both generator branches use (np.unique /
            # np.arange), so the column order of the generated PSU matrix
            # matches unit_to_psu_col.
            if resolved_survey_unit.psu is not None:
                psu_ids = np.unique(resolved_survey_unit.psu)
            else:
                psu_ids = np.arange(len(resolved_survey_unit.weights))
            if len(psu_ids) < 2:
                import warnings as _warnings

                _warnings.warn(
                    f"{self._BOOTSTRAP_LABEL} bootstrap with survey/cluster design "
                    f"has only {len(psu_ids)} PSU(s); bootstrap variance is "
                    "unidentified. All bootstrap inference fields "
                    "(overall_se, group_time_ses, event_study_ses, "
                    "group_effect_ses, and their CIs / p-values) will be "
                    "NaN. Use n_bootstrap=0 (analytical IF variance) or "
                    "a design with at least 2 PSUs.",
                    UserWarning,
                    stacklevel=2,
                )
                _bootstrap_cluster_variance_unidentified = True
            # Build unit → PSU column map
            if resolved_survey_unit.psu is not None:
                unit_psu = resolved_survey_unit.psu
                psu_id_to_col = {int(p): c for c, p in enumerate(psu_ids)}
                unit_to_psu_col = np.array(
                    [psu_id_to_col[int(unit_psu[i])] for i in range(n_units)]
                )
            else:
                # Each unit is its own PSU — identity mapping
                unit_to_psu_col = np.arange(n_units)

            # When each unit is its own PSU (e.g. cluster="unit"), the PSU block
            # is already unit-aligned, so the fancy-index expansion is an
            # identity permutation whose only effect is a needless full-block
            # copy (doubling live block memory). Detect that once and skip it.
            _psu_is_identity = len(psu_ids) == n_units and bool(
                np.array_equal(unit_to_psu_col, np.arange(n_units))
            )

            # Factory recreating the PSU generation + unit-level expansion per
            # pass; the full (n_bootstrap, n_units) expansion is never
            # materialized at once.
            def _make_weight_iter(
                rng_: np.random.Generator,
            ) -> Iterator[Tuple[int, np.ndarray]]:
                _, _psu_blocks = iter_survey_multiplier_weight_blocks(
                    self.n_bootstrap,
                    resolved_survey_unit,
                    self.bootstrap_weights,
                    rng_,
                    block_size=_block_size,
                )

                def _expanded() -> Iterator[Tuple[int, np.ndarray]]:
                    for _cs, _psu_block in _psu_blocks:
                        if _psu_is_identity:
                            yield _cs, _psu_block
                        else:
                            yield _cs, _psu_block[:, unit_to_psu_col]

                return _expanded()

        else:
            # Standard unit-level weights (no survey or weights-only), generated
            # one row-block at a time directly at unit width.
            def _make_weight_iter(
                rng_: np.random.Generator,
            ) -> Iterator[Tuple[int, np.ndarray]]:
                return iter_weight_blocks(self.n_bootstrap, n_units, self.bootstrap_weights, rng_)

        # Snapshot the rng state HERE — the value that fully determines the
        # weight stream (nothing above consumed the rng; the survey psu
        # resolution deliberately avoids it). Retained on the returned
        # container for the post-fit aggregate() replay. `_replay_backend`
        # records the generation-branch identity: "portable" for branches
        # whose draws are provably identical under either weight backend —
        # the stratified / single-PSU survey generator draws through the
        # NumPy generator unconditionally, and the unstratified census-FPC
        # case (fpc[0] <= n_psu, mirroring iter_survey_multiplier_weight_
        # blocks' fpc_zero) replaces every block with zeros — else the
        # current effective backend, because Rust and NumPy produce
        # DIFFERENT draws from the same bit-generator state.
        replay_bitgen_state = dict(rng.bit_generator.state)
        _backend_independent = False
        if _use_survey_bootstrap:
            assert resolved_survey_unit is not None
            _fpc = getattr(resolved_survey_unit, "fpc", None)
            _n_psu = len(psu_ids)  # bound above in the survey branch
            _backend_independent = (
                resolved_survey_unit.strata is not None
                or _n_psu < 2
                or (_fpc is not None and _n_psu / _fpc[0] >= 1.0)
            )
        replay_backend = "portable" if _backend_independent else effective_weight_backend()

        # Re-iterable stream: each column tile of the fused perturbation GEMM
        # below makes its own full pass over the bit-identical weight stream.
        weight_stream = ReplayableWeightStream(_make_weight_iter, rng)

        # Pre-compute the overall combined IF once (reused across every block).
        # None exactly when the overall aggregation is skipped.
        overall_combined_if: Optional[np.ndarray] = None
        if not skip_overall_aggregation:
            # Use combined IF (standard IF + WIF) for proper bootstrap
            post_gt_pairs = [gt_pairs[i] for i in post_treatment_indices]
            post_groups = np.array([gt_pairs[i][0] for i in post_treatment_indices])
            post_effects = original_atts[post_treatment_indices]
            overall_combined_if, _ = self._compute_combined_influence_function(
                post_gt_pairs,
                overall_weights_post,
                post_effects,
                post_groups,
                influence_func_info,
                df,
                unit,
                precomputed,
                global_unit_to_idx=unit_to_idx,
                n_global_units=n_units,
            )

        rel_periods: List[int] = []
        if event_study_info is not None:
            rel_periods = sorted(event_study_info.keys())

        group_list: List[Any] = []
        if group_agg_info is not None:
            group_list = sorted(group_agg_info.keys())

        # Fused perturbation columns: [per-cell IFs | overall combined IF |
        # per-event-time combined IFs]. One column-tiled GEMM over the
        # replayable weight stream replaces the former per-cell
        # ``W[:, idx] @ inf`` slicing loop, which was memory-bandwidth-bound
        # (two fancy-index copies of the weight block per cell). The weight
        # stream is bit-identical to the per-block path; the BLAS reductions
        # may reassociate, so statistics match to within ~1 ULP (far below
        # bootstrap Monte-Carlo error), not bit-for-bit. Treated/control index
        # arrays are disjoint per cell, satisfying the kernel's
        # assignment-scatter contract.
        columns: List[Any] = [
            [
                (gt_treated_indices[j], gt_treated_inf[j]),
                (gt_control_indices[j], gt_control_inf[j]),
            ]
            for j in range(n_gt)
        ]
        overall_col = -1
        if overall_combined_if is not None:
            overall_col = len(columns)
            columns.append([(None, overall_combined_if)])
        es_col0 = len(columns)
        # rel_periods is non-empty only when event-study info was built.
        assert event_study_info is not None or not rel_periods
        for e in rel_periods:
            assert event_study_info is not None
            columns.append([(None, event_study_info[e]["combined_if"])])

        perturbations = tiled_if_matmul(weight_stream, self.n_bootstrap, n_units, columns)

        # Reconstruct the bootstrap draws (small, n_bootstrap-sized arrays).
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            bootstrap_atts_gt = original_atts[None, :] + perturbations[:, :n_gt]
            if skip_overall_aggregation:
                bootstrap_overall = np.full(self.n_bootstrap, np.nan)
            else:
                bootstrap_overall = original_overall + perturbations[:, overall_col]

            bootstrap_event_study: Optional[Dict[int, np.ndarray]] = None
            if event_study_info is not None:
                bootstrap_event_study = {
                    e: event_study_info[e]["effect"] + perturbations[:, es_col0 + k]
                    for k, e in enumerate(rel_periods)
                }

            # Group aggregation: fixed-weight re-aggregation of the completed
            # perturbed cell draws (matches at reassociation level).
            bootstrap_group: Optional[Dict[Any, np.ndarray]] = None
            if group_agg_info is not None:
                bootstrap_group = {
                    g: bootstrap_atts_gt[:, group_agg_info[g]["gt_indices"]]
                    @ group_agg_info[g]["weights"]
                    for g in group_list
                }

        # Batch compute bootstrap statistics for ATT(g,t)
        batch_ses, batch_ci_lo, batch_ci_hi, batch_pv = _compute_effect_bootstrap_stats_batch_func(
            original_atts,
            bootstrap_atts_gt,
            alpha=self.alpha,
        )
        gt_ses = {}
        gt_cis = {}
        gt_p_values = {}
        for j, gt in enumerate(gt_pairs):
            gt_ses[gt] = float(batch_ses[j])
            gt_cis[gt] = (float(batch_ci_lo[j]), float(batch_ci_hi[j]))
            gt_p_values[gt] = float(batch_pv[j])

        # Compute bootstrap statistics for overall ATT
        if skip_overall_aggregation:
            overall_se = np.nan
            overall_ci = (np.nan, np.nan)
            overall_p_value = np.nan
        else:
            overall_se, overall_ci, overall_p_value = _compute_effect_bootstrap_stats_func(
                original_overall,
                bootstrap_overall,
                alpha=self.alpha,
                context="overall ATT",
            )

        # Batch compute bootstrap statistics for event study effects
        event_study_ses = None
        event_study_cis = None
        event_study_p_values = None
        # Paper Eq. (4.14) overall (event-study average) bootstrap inference; stays
        # NaN unless event-study draws exist. Mirrors the analytical overall_att_es.
        overall_att_es_se = np.nan
        overall_att_es_ci = (np.nan, np.nan)
        overall_att_es_p_value = np.nan

        # ``rel_periods`` can be empty when balance_e (or an empty event study) leaves
        # no relative periods; guard the column_stack so the bootstrap mirrors the
        # analytical empty/NaN surface instead of raising "need at least one array to
        # concatenate". event_study_ses stays None and the Eq. 4.14 overall bootstrap
        # stays NaN (the analytical fit path emits the requested-but-undefined warning).
        if bootstrap_event_study is not None and event_study_info is not None and rel_periods:
            es_effects = np.array([event_study_info[e]["effect"] for e in rel_periods])
            es_boot_matrix = np.column_stack([bootstrap_event_study[e] for e in rel_periods])
            es_ses, es_ci_lo, es_ci_hi, es_pv = _compute_effect_bootstrap_stats_batch_func(
                es_effects,
                es_boot_matrix,
                alpha=self.alpha,
            )
            event_study_ses = {e: float(es_ses[i]) for i, e in enumerate(rel_periods)}
            event_study_cis = {
                e: (float(es_ci_lo[i]), float(es_ci_hi[i])) for i, e in enumerate(rel_periods)
            }
            event_study_p_values = {e: float(es_pv[i]) for i, e in enumerate(rel_periods)}

            # Eq. (4.14) overall = unweighted mean of post-treatment ES(e) (e >=
            # -anticipation, matching post_treatment_mask above). The per-draw mean
            # over those event times is the bootstrap distribution of the overall.
            es_post = [
                e
                for e in rel_periods
                if e >= -self.anticipation
                and e in event_study_info
                and np.isfinite(event_study_info[e]["effect"])
            ]
            if es_post:
                original_es_overall = float(
                    np.mean([event_study_info[e]["effect"] for e in es_post])
                )
                boot_es_overall = np.column_stack([bootstrap_event_study[e] for e in es_post]).mean(
                    axis=1
                )
                (
                    overall_att_es_se,
                    overall_att_es_ci,
                    overall_att_es_p_value,
                ) = _compute_effect_bootstrap_stats_func(
                    original_es_overall,
                    boot_es_overall,
                    alpha=self.alpha,
                    context="overall ATT (event-study average)",
                )

        # Batch compute bootstrap statistics for group effects
        group_effect_ses = None
        group_effect_cis = None
        group_effect_p_values = None

        # ``group_list`` can be EMPTY when no cohort has a post-treatment cell
        # (e.g. every treated cohort's onset lies beyond the observed panel):
        # np.column_stack([]) would raise "need at least one array to
        # concatenate", so guard it exactly as the event-study block guards
        # empty ``rel_periods`` above — the group stats stay None and the
        # aggregation returns its supported zero-row result.
        if bootstrap_group is not None and group_agg_info is not None and group_list:
            grp_effects = np.array([group_agg_info[g]["effect"] for g in group_list])
            grp_boot_matrix = np.column_stack([bootstrap_group[g] for g in group_list])
            grp_ses, grp_ci_lo, grp_ci_hi, grp_pv = _compute_effect_bootstrap_stats_batch_func(
                grp_effects,
                grp_boot_matrix,
                alpha=self.alpha,
            )
            group_effect_ses = {g: float(grp_ses[i]) for i, g in enumerate(group_list)}
            group_effect_cis = {
                g: (float(grp_ci_lo[i]), float(grp_ci_hi[i])) for i, g in enumerate(group_list)
            }
            group_effect_p_values = {g: float(grp_pv[i]) for i, g in enumerate(group_list)}

        # Compute simultaneous confidence band critical value (sup-t)
        cband_crit_value = None
        if (
            cband
            and bootstrap_event_study is not None
            and event_study_ses is not None
            and event_study_info is not None
        ):
            valid_es = [
                e
                for e in rel_periods
                if e in event_study_ses
                and np.isfinite(event_study_ses[e])
                and event_study_ses[e] > 0
            ]
            if valid_es:
                # Vectorized sup_t: max_e |(boot_att_e[b] - att_e) / se_e|
                boot_matrix = np.array([bootstrap_event_study[e] for e in valid_es])
                effects_vec = np.array([event_study_info[e]["effect"] for e in valid_es])
                ses_vec = np.array([event_study_ses[e] for e in valid_es])
                with np.errstate(divide="ignore", invalid="ignore"):
                    sup_t_dist = np.max(
                        np.abs((boot_matrix - effects_vec[:, None]) / ses_vec[:, None]),
                        axis=0,
                    )
                finite_mask = np.isfinite(sup_t_dist)
                n_valid = int(np.sum(finite_mask))
                n_total = len(sup_t_dist)
                if n_valid < n_total * 0.5:
                    warnings.warn(
                        f"Too few valid sup-t bootstrap samples ({n_valid}/{n_total}). "
                        "Returning None for cband critical value.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                elif n_valid > 0:
                    cband_crit_value = float(np.quantile(sup_t_dist[finite_mask], 1 - self.alpha))

        # NaN-out all bootstrap inference surfaces when clustered
        # bootstrap variance is unidentified (G<2 PSUs). See guard
        # added at the top of the bootstrap weight generation.
        if _bootstrap_cluster_variance_unidentified:
            overall_se = np.nan
            overall_ci = (np.nan, np.nan)
            overall_p_value = np.nan
            overall_att_es_se = np.nan
            overall_att_es_ci = (np.nan, np.nan)
            overall_att_es_p_value = np.nan
            gt_ses = {gt: np.nan for gt in gt_ses} if gt_ses else gt_ses
            gt_cis = {gt: (np.nan, np.nan) for gt in gt_cis} if gt_cis else gt_cis
            gt_p_values = {gt: np.nan for gt in gt_p_values} if gt_p_values else gt_p_values
            if event_study_ses:
                # ses/cis/p_values are populated together upstream.
                assert event_study_cis is not None and event_study_p_values is not None
                event_study_ses = {k: np.nan for k in event_study_ses}
                event_study_cis = {k: (np.nan, np.nan) for k in event_study_cis}
                event_study_p_values = {k: np.nan for k in event_study_p_values}
            if group_effect_ses:
                assert group_effect_cis is not None and group_effect_p_values is not None
                group_effect_ses = {k: np.nan for k in group_effect_ses}
                group_effect_cis = {k: (np.nan, np.nan) for k in group_effect_cis}
                group_effect_p_values = {k: np.nan for k in group_effect_p_values}
            cband_crit_value = None

        result = CSBootstrapResults(
            n_bootstrap=self.n_bootstrap,
            weight_type=self.bootstrap_weights,
            alpha=self.alpha,
            overall_att_se=overall_se,
            overall_att_ci=overall_ci,
            overall_att_p_value=overall_p_value,
            group_time_ses=gt_ses,
            group_time_cis=gt_cis,
            group_time_p_values=gt_p_values,
            event_study_ses=event_study_ses,
            event_study_cis=event_study_cis,
            event_study_p_values=event_study_p_values,
            group_effect_ses=group_effect_ses,
            group_effect_cis=group_effect_cis,
            group_effect_p_values=group_effect_p_values,
            bootstrap_distribution=bootstrap_overall,
            cband_crit_value=cband_crit_value,
            overall_att_es_se=overall_att_es_se,
            overall_att_es_ci=overall_att_es_ci,
            overall_att_es_p_value=overall_att_es_p_value,
        )
        result._replay_bitgen_state = replay_bitgen_state
        result._replay_backend = replay_backend
        return result

    def _prepare_event_study_aggregation(
        self,
        gt_pairs: List[Tuple[Any, Any]],
        group_time_effects: Dict,
        balance_e: Optional[int],
        influence_func_info: Any = None,
        df: Any = None,
        unit: Optional[str] = None,
        precomputed: Any = None,
        global_unit_to_idx: Optional[Dict[Any, int]] = None,
        n_global_units: Optional[int] = None,
    ) -> Dict[int, Dict[str, Any]]:
        """Prepare aggregation info for event study bootstrap."""
        # Use fixed cohort survey masses (not per-cell survey_weight_sum) when
        # survey weights are present, matching the analytical
        # _aggregate_event_study() path.
        # Shared fixed-cohort masses (same source of truth as the analytical
        # event-study path): unit-level RC mass preferred so the bootstrap
        # event-study weights an unbalanced-panel (allow_unbalanced_panel)
        # aggregation by fixed UNIT cohort mass, not observation count.
        from diff_diff.staggered_aggregation import fixed_cohort_agg_weights

        _fixed_masses = fixed_cohort_agg_weights(precomputed)

        def _agg_weight(g: Any, t: Any) -> float:
            if _fixed_masses is not None and g in _fixed_masses:
                return _fixed_masses[g]
            # Use agg_weight if available (RCS: fixed cohort mass)
            return group_time_effects[(g, t)].get(
                "agg_weight", group_time_effects[(g, t)]["n_treated"]
            )

        # Organize by relative time
        effects_by_e: Dict[int, List[Tuple[int, float, float]]] = {}

        for j, (g, t) in enumerate(gt_pairs):
            e = t - g
            if e not in effects_by_e:
                effects_by_e[e] = []
            effects_by_e[e].append(
                (
                    j,  # index in gt_pairs
                    group_time_effects[(g, t)]["effect"],
                    _agg_weight(g, t),
                )
            )

        # Balance if requested
        if balance_e is not None:
            groups_at_e = set()
            for j, (g, t) in enumerate(gt_pairs):
                if t - g == balance_e and np.isfinite(group_time_effects[(g, t)]["effect"]):
                    groups_at_e.add(g)

            balanced_effects: Dict[int, List[Tuple[int, float, float]]] = {}
            for j, (g, t) in enumerate(gt_pairs):
                if g in groups_at_e:
                    e = t - g
                    if e not in balanced_effects:
                        balanced_effects[e] = []
                    balanced_effects[e].append(
                        (
                            j,
                            group_time_effects[(g, t)]["effect"],
                            _agg_weight(g, t),
                        )
                    )
            effects_by_e = balanced_effects

        # Compute aggregation weights
        result = {}
        for e, effect_list in effects_by_e.items():
            indices = np.array([x[0] for x in effect_list])
            effects = np.array([x[1] for x in effect_list])
            n_treated = np.array([x[2] for x in effect_list], dtype=float)

            # Exclude NaN effects (matches analytical aggregation path)
            finite_mask = np.isfinite(effects)
            if not np.all(finite_mask):
                indices = indices[finite_mask]
                effects = effects[finite_mask]
                n_treated = n_treated[finite_mask]
                if len(effects) == 0:
                    continue

            weights = n_treated / np.sum(n_treated)
            agg_effect = np.sum(weights * effects)

            entry: Dict[str, Any] = {
                "gt_indices": indices,
                "weights": weights,
                "effect": agg_effect,
            }

            # Compute combined IF for this event time if args available.
            # `precomputed` alone suffices for the in-package fast path
            # (kit-backed post-fit replay threads df=None/unit=None); the
            # df/unit pair remains for direct callers without precomputed.
            if influence_func_info is not None and (
                precomputed is not None or (df is not None and unit is not None)
            ):
                gt_pairs_for_e = [gt_pairs[i] for i in indices]
                groups_for_gt = np.array([gt_pairs[i][0] for i in indices])
                combined_if, _ = self._compute_combined_influence_function(
                    gt_pairs_for_e,
                    weights,
                    effects,
                    groups_for_gt,
                    influence_func_info,
                    df,
                    unit,
                    precomputed,
                    global_unit_to_idx=global_unit_to_idx,
                    n_global_units=n_global_units,
                )
                entry["combined_if"] = combined_if

            result[e] = entry

        return result

    def _prepare_group_aggregation(
        self,
        gt_pairs: List[Tuple[Any, Any]],
        group_time_effects: Dict,
        treatment_groups: List[Any],
    ) -> Dict[Any, Dict[str, Any]]:
        """Prepare aggregation info for group-level bootstrap."""
        result = {}

        for g in treatment_groups:
            # Get all effects for this group (post-treatment only: t >= g - anticipation)
            group_data = []
            for j, (gg, t) in enumerate(gt_pairs):
                if gg == g and t >= g - self.anticipation:
                    group_data.append(
                        (
                            j,
                            group_time_effects[(gg, t)]["effect"],
                        )
                    )

            if not group_data:
                continue

            indices = np.array([x[0] for x in group_data])
            effects = np.array([x[1] for x in group_data])

            # Exclude NaN effects (matches analytical aggregation path)
            finite_mask = np.isfinite(effects)
            if not np.all(finite_mask):
                indices = indices[finite_mask]
                effects = effects[finite_mask]
                if len(effects) == 0:
                    continue

            # Equal weights across time periods
            weights = np.ones(len(effects)) / len(effects)
            agg_effect = np.sum(weights * effects)

            result[g] = {
                "gt_indices": indices,
                "weights": weights,
                "effect": agg_effect,
            }

        return result

    def _compute_percentile_ci(
        self,
        boot_dist: np.ndarray,
        alpha: float,
    ) -> Tuple[float, float]:
        """Compute percentile confidence interval from bootstrap distribution."""
        return _compute_percentile_ci_func(boot_dist, alpha)

    def _compute_bootstrap_pvalue(
        self,
        original_effect: float,
        boot_dist: np.ndarray,
        n_valid: Optional[int] = None,
    ) -> float:
        """
        Compute two-sided bootstrap p-value.

        Delegates to :func:`bootstrap_utils.compute_bootstrap_pvalue`.
        """
        return _compute_bootstrap_pvalue_func(original_effect, boot_dist, n_valid=n_valid)

    def _compute_effect_bootstrap_stats(
        self,
        original_effect: float,
        boot_dist: np.ndarray,
        context: str = "bootstrap distribution",
    ) -> Tuple[float, Tuple[float, float], float]:
        """
        Compute bootstrap statistics for a single effect.

        Delegates to :func:`bootstrap_utils.compute_effect_bootstrap_stats`.
        """
        return _compute_effect_bootstrap_stats_func(
            original_effect, boot_dist, alpha=self.alpha, context=context
        )


# =============================================================================
# Bootstrap override helpers (shared by fit and the post-fit replay)
# =============================================================================
# Extracted verbatim from CallawaySantAnna.fit()'s inline blocks so the
# post-fit aggregate() replay applies EXACTLY the same percentile overrides
# the fit-time path applies — one implementation, no twin drift. (The
# deprecated StaggeredTripleDifference keeps its OWN copy of the group
# replacement loop; unifying it is sequenced with the M-014 container port.)
# Note on warning attribution: when these run under the post-fit replay the
# engine's fit-tuned stacklevels resolve into library frames rather than the
# user's aggregate() call — accepted as cosmetic (recorded decision).


def apply_bootstrap_event_study_overrides(
    event_study_effects: Optional[Dict[int, Dict[str, Any]]],
    bootstrap_results: CSBootstrapResults,
    alpha: float,
) -> None:
    """Overwrite per-event-time se/CI/p with percentile-bootstrap values.

    Mutates ``event_study_effects`` in place; t is recomputed from the
    percentile SE via ``safe_inference_batch``. No-op when either side has
    no event-study surface.
    """
    if (
        event_study_effects is not None
        and bootstrap_results.event_study_ses is not None
        and bootstrap_results.event_study_cis is not None
        and bootstrap_results.event_study_p_values is not None
    ):
        es_keys = [e for e in event_study_effects if e in bootstrap_results.event_study_ses]
        if es_keys:
            es_effects_arr = np.array([float(event_study_effects[e]["effect"]) for e in es_keys])
            es_ses_arr = np.array([float(bootstrap_results.event_study_ses[e]) for e in es_keys])
            es_t_stats, _, _, _ = safe_inference_batch(es_effects_arr, es_ses_arr, alpha=alpha)
            for idx, e in enumerate(es_keys):
                event_study_effects[e]["se"] = bootstrap_results.event_study_ses[e]
                event_study_effects[e]["conf_int"] = bootstrap_results.event_study_cis[e]
                event_study_effects[e]["p_value"] = bootstrap_results.event_study_p_values[e]
                event_study_effects[e]["t_stat"] = float(es_t_stats[idx])


def apply_bootstrap_group_overrides(
    group_effects: Optional[Dict[Any, Dict[str, Any]]],
    bootstrap_results: CSBootstrapResults,
    alpha: float,
) -> None:
    """Overwrite per-group se/CI/p with percentile-bootstrap values.

    Mutates ``group_effects`` in place and clears each row's ``df_used``
    (the percentile inference never used the analytical df, so keeping it
    would claim a t-reference that governed nothing). No-op when either
    side has no group surface.
    """
    if (
        group_effects is not None
        and bootstrap_results.group_effect_ses is not None
        and bootstrap_results.group_effect_cis is not None
        and bootstrap_results.group_effect_p_values is not None
    ):
        grp_keys = [g for g in group_effects if g in bootstrap_results.group_effect_ses]
        if grp_keys:
            grp_effects_arr = np.array([float(group_effects[g]["effect"]) for g in grp_keys])
            grp_ses_arr = np.array([float(bootstrap_results.group_effect_ses[g]) for g in grp_keys])
            grp_t_stats, _, _, _ = safe_inference_batch(grp_effects_arr, grp_ses_arr, alpha=alpha)
            for idx, g in enumerate(grp_keys):
                group_effects[g]["se"] = bootstrap_results.group_effect_ses[g]
                group_effects[g]["conf_int"] = bootstrap_results.group_effect_cis[g]
                group_effects[g]["p_value"] = bootstrap_results.group_effect_p_values[g]
                group_effects[g]["t_stat"] = float(grp_t_stats[idx])
                # Same clearing rule the ES df provenance follows: these
                # se/p/CI are now percentile-bootstrap values that never used
                # the analytical df, so keeping df_used would claim a
                # t-reference that governed nothing.
                group_effects[g]["df_used"] = None


def apply_cband_conf_ints(
    event_study_effects: Optional[Dict[int, Dict[str, Any]]],
    cband_crit_value: Optional[float],
) -> None:
    """Attach simultaneous-band CIs per event time from the sup-t critical value.

    Mutates ``event_study_effects`` in place; no-op when the critical value
    or the surface is absent.
    """
    if cband_crit_value is not None and event_study_effects is not None:
        for _e, eff_data in event_study_effects.items():
            se_val = eff_data["se"]
            if np.isfinite(se_val) and se_val > 0:
                eff_data["cband_conf_int"] = (
                    eff_data["effect"] - cband_crit_value * se_val,
                    eff_data["effect"] + cband_crit_value * se_val,
                )
