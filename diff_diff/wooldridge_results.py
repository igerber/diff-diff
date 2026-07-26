"""Results class for WooldridgeDiD (ETWFE) estimator."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from diff_diff.results_base import BaseResults
from diff_diff.utils import safe_inference


@dataclass
class WooldridgeDiDResults(BaseResults):
    """Results from WooldridgeDiD.fit().

    Core output is ``group_time_effects``: a dict keyed by (cohort_g, time_t)
    with per-cell ATT estimates and inference. Call
    ``.aggregate(type, weights=...)`` to compute any of the four
    ``jwdid_estat`` aggregation types under either the default
    cell-count weighting (``weights="cell"``, matches Stata
    ``jwdid_estat``) or the paper W2025 opt-in cohort-share weighting
    (``weights="cohort_share"``, Eqs. 7.4 / 7.6; restricted to
    ``type ∈ {"simple", "event"}``). ``cohort_trend_coefs`` carries
    Section 8 / Eq. 8.1 estimated ``δ_g`` slopes when the fit was
    produced under ``WooldridgeDiD(cohort_trends=True)``.
    ``aggregation_weights`` is keyed by aggregation type and records
    the active weighting scheme that wrote to each cached surface
    (surfaced in ``summary()`` / ``to_dataframe()`` / ``__repr__``).
    """

    # ------------------------------------------------------------------ #
    # Core cohort×time estimates                                          #
    # ------------------------------------------------------------------ #
    group_time_effects: Dict[Tuple[Any, Any], Dict[str, Any]]
    """key=(g,t), value={att, se, t_stat, p_value, conf_int}"""

    # ------------------------------------------------------------------ #
    # Simple (overall) aggregation — always populated at fit time         #
    # ------------------------------------------------------------------ #
    overall_att: float
    overall_se: float
    overall_t_stat: float
    overall_p_value: float
    overall_conf_int: Tuple[float, float]

    # ------------------------------------------------------------------ #
    # Other aggregations — populated by .aggregate()                      #
    # ------------------------------------------------------------------ #
    group_effects: Optional[Dict[Any, Dict]] = field(default=None, repr=False)
    calendar_effects: Optional[Dict[Any, Dict]] = field(default=None, repr=False)
    event_study_effects: Optional[Dict[int, Dict]] = field(default=None, repr=False)

    # ------------------------------------------------------------------ #
    # Metadata                                                            #
    # ------------------------------------------------------------------ #
    method: str = "ols"
    control_group: str = "not_yet_treated"
    groups: List[Any] = field(default_factory=list)
    time_periods: List[Any] = field(default_factory=list)
    n_obs: int = 0
    n_treated_units: int = 0
    n_control_units: int = 0
    alpha: float = 0.05
    anticipation: int = 0
    survey_metadata: Optional[Any] = field(default=None, repr=False)

    # Variance-family metadata. ``vcov_type`` records the configured analytical
    # family ("classical", "hc1", "hc2", "hc2_bm", or "conley" — the conley
    # spatial-HAC path also populates ``conley_lag_cutoff``); when ``survey_design=``
    # is supplied the survey TSL (or replicate-weight refit) variance overrides
    # this — the field still records the configured value and
    # ``survey_metadata`` indicates the survey path was active. On bootstrap
    # fits (``n_bootstrap > 0``) the SE comes from the multiplier bootstrap,
    # not the analytical family. ``cluster_name`` / ``n_clusters`` are
    # populated when the fit was clustered (default unit cluster, or
    # user-set ``cluster=X``); both are ``None`` on explicit one-way
    # (``vcov_type in {"classical","hc2"}`` + no user cluster) fits where
    # the auto-cluster was dropped.
    vcov_type: str = "hc1"
    cluster_name: Optional[str] = None
    n_clusters: Optional[int] = None
    # Conley spatial-HAC within-unit Bartlett max lag (populated only when
    # ``vcov_type == "conley"``; ``None`` otherwise). Carries the configured
    # ``conley_lag_cutoff`` for the summary variance label.
    conley_lag_cutoff: Optional[int] = None

    # Heterogeneous cohort-specific linear trends (paper W2025 Section 8 /
    # Eq. 8.1). Keyed by treated cohort ``g`` → estimated slope ``δ_g``.
    # Empty dict when ``WooldridgeDiD`` was fit with ``cohort_trends=False``
    # (the default). Populated only via the OLS path; logit / poisson
    # reject ``cohort_trends=True`` at the constructor per paper Section 8
    # OLS-only scope.
    #
    # Identification + baseline normalization (paper W2025 Section 5.4):
    # the reported ``δ_g`` slopes are RELATIVE TO THE BASELINE TREND
    # absorbed by the design — the never-treated cohort's trend (when a
    # never-treated cohort exists) OR the last cohort's trend (when no
    # never-treated cohort exists, per the all-eventually-treated drop
    # rule). On all-treated panels the last cohort is intentionally
    # absent from the dict; its slope is the baseline (zero in deviation
    # form). The all-treated branch is currently UNREACHABLE: the paper
    # rule is applied to the trend columns but not to the cohort × time
    # cells, so those panels raise at fit() (TODO.md). See REGISTRY
    # ``## WooldridgeDiD (ETWFE)`` → "Heterogeneous cohort trends" Notes
    # for the exact normalization contract.
    cohort_trend_coefs: Dict[Any, float] = field(default_factory=dict, repr=False)

    # Flag set by ``_fit_ols`` when ``n_bootstrap > 0`` AND the multiplier
    # bootstrap actually ran (i.e., produced at least one valid bootstrap
    # statistic). When True, ``aggregate(type="simple", weights="cell")``
    # is a no-op (preserves the bootstrap inference populated at fit time)
    # and ``aggregate(type="simple", weights="cohort_share")`` raises
    # because the cohort-share aggregation is not bootstrapped — re-fit
    # with ``n_bootstrap=0`` to use cohort-share + analytical inference,
    # or wait for the deferred bootstrap-cohort-share follow-up.
    _bootstrap_used: bool = field(default=False, repr=False)

    # Model-surface metadata for self-describing reporting.
    # ``cohort_trends`` records whether the fit was produced under the
    # Section 8 / Eq. 8.1 heterogeneous-cohort-trends design (paper
    # W2025 ``dg_i · t`` interactions on the OLS path). False on the
    # default ``cohort_trends=False`` fit and on logit/Poisson paths
    # (which reject ``cohort_trends=True`` at the constructor).
    #
    # ``aggregation_weights`` records the weighting scheme PER cached
    # aggregation surface so ``summary()`` / ``to_dataframe()`` /
    # ``__repr__()`` can label each surface correctly under mixed-order
    # ``aggregate(weights=...)`` calls. Keys: ``"simple"`` (matches the
    # ``overall_*`` fields), ``"group"``, ``"calendar"``, ``"event"``.
    # The fit-time ``overall_*`` is cell-weighted, so ``"simple"`` is
    # initialized to ``"cell"`` and only flips after a successful
    # ``aggregate(type="simple", weights="cohort_share")`` call. The
    # other keys are populated lazily by ``aggregate()``. Mutation is
    # atomic — only set after the aggregation passes all validation
    # AND completes successfully, so failed cohort_share calls on
    # survey-weighted or bootstrap fits leave metadata unchanged
    # (codex CI R7 P1 fix).
    cohort_trends: bool = field(default=False, repr=False)
    aggregation_weights: Dict[str, str] = field(
        default_factory=lambda: {"simple": "cell"}, repr=False
    )

    # ------------------------------------------------------------------ #
    # Internal — used by aggregate() for delta-method SEs                 #
    # ------------------------------------------------------------------ #
    _gt_weights: Dict[Tuple[Any, Any], int] = field(default_factory=dict, repr=False)
    _n_g_per_cohort: Dict[Any, int] = field(default_factory=dict, repr=False)
    """Unit count per treated cohort ``g`` (``N_g`` in paper Eqs. 7.4, 7.6).
    Populated at fit time from the analysis sample; used by
    ``aggregate(weights="cohort_share")`` (paper Section 7) to compute
    the simple-overall cohort-share weights ``ω̂_g`` and event-time
    weights ``ω̂_{ge}``. Empty dict on fits that pre-date the PR-B
    cohort-share surface (no information loss — ``weights="cell"`` is
    unaffected)."""
    _gt_vcov: Optional[np.ndarray] = field(default=None, repr=False)
    """Full vcov of all β_{g,t} coefficients (ordered same as sorted group_time_effects keys)."""
    _gt_keys: List[Tuple[Any, Any]] = field(default_factory=list, repr=False)
    """Ordered list of (g,t) keys corresponding to _gt_vcov columns."""
    _df_survey: Optional[int] = field(default=None, repr=False)
    """Survey degrees of freedom for t-distribution inference."""
    _bm_per_cell_dof: Dict[Tuple[Any, Any], float] = field(default_factory=dict, repr=False)
    """Per-cell Bell-McCaffrey Satterthwaite DOF (only populated for vcov_type='hc2_bm').
    Used by group_time_effects[(g, t)] inference fields at fit time."""
    _bm_artifacts: Optional[
        Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[Tuple[Any, Any], int]]
    ] = field(default=None, repr=False)
    """(X_red, cluster_ids, bread_red, coef_idx_map) for hc2_bm; enables
    lazy BM contrast-DOF computation in aggregate().

    ``X_red`` / ``bread_red`` are the REDUCED (kept-column) design and bread
    matrix produced by ``_fit_ols`` after rank-deficient column drops — the
    same subspace ``solve_ols`` returned non-NaN coefficients in.
    ``coef_idx_map`` maps each ``(g, t)`` cell present in
    ``group_time_effects`` to its column index in ``X_red``. Storing reduced
    artifacts avoids the singular full-design bread that
    ``_compute_cr2_bm_contrast_dof`` would otherwise reject."""
    _df_one_way: Optional[float] = field(default=None, repr=False)
    """Residual DOF (``n - rank(X)``) for one-way ``vcov_type in
    {"classical","hc2"}`` paths (full-dummy, no survey). ``aggregate()``
    uses this to thread R's ``lm()`` t-distribution into per-key
    inference. ``None`` on hc1 / hc2_bm / surveyed paths (which use BM
    DOF or ``_df_survey`` instead)."""

    # ------------------------------------------------------------------ #
    # Public methods                                                      #
    # ------------------------------------------------------------------ #

    def aggregate(self, type: str, weights: str = "cell") -> "WooldridgeDiDResults":  # noqa: A002
        """Compute and store one of the four jwdid_estat aggregation types.

        Parameters
        ----------
        type : "simple" | "group" | "calendar" | "event"
        weights : "cell" | "cohort_share", default "cell"
            Aggregation weighting scheme. ``"cell"`` (default) uses cell-
            count ``n_{g,t}`` observation counts and matches Stata
            ``jwdid_estat``. ``"cohort_share"`` uses paper W2025 Eq. 7.4
            ``ω̂_g = N_g / Σ_{g'} N_{g'} M_{g'}`` for ``type="simple"`` and
            Eq. 7.6 ``ω̂_{ge} = N_g / Σ_{g': g'+e ≤ T} N_{g'}`` for
            ``type="event"``. Both formulas reduce to ``N_g``-proportional
            per-cell weights with the appropriate normalization. The two
            schemes coincide on balanced panels with uniform within-cohort
            cell counts (paper Section 7.5). The cohort-share scheme is
            supported only for ``type="simple"`` and ``type="event"``; the
            paper provides no explicit cohort-share formula for ``"group"``
            or ``"calendar"`` aggregations and the library raises
            ``ValueError`` to preserve a fail-closed contract.

        Returns self for chaining.

        Notes
        -----
        When ``vcov_type == "hc2_bm"``, aggregated inference (t_stat / p_value /
        conf_int) uses Bell-McCaffrey Satterthwaite contrast-specific DOFs
        rather than the survey/None default. The BM DOFs are computed lazily
        from ``_bm_artifacts`` via ``_compute_cr2_bm_contrast_dof`` and
        fail-closed (NaN inference) when the helper raises or returns NaN —
        per ``feedback_bm_contrast_dof_fail_closed``. The contrast column
        is rebuilt under the active ``weights`` scheme so the BM DOF
        reflects the actual weighting used by ATT + SE.
        """
        valid = ("simple", "group", "calendar", "event")
        if type not in valid:
            raise ValueError(f"type must be one of {valid}, got {type!r}")

        valid_weights = ("cell", "cohort_share")
        if weights not in valid_weights:
            raise ValueError(f"weights must be one of {valid_weights}, got {weights!r}")
        if weights == "cohort_share" and type in ("group", "calendar"):
            raise ValueError(
                f"weights='cohort_share' is only supported for type='simple' "
                f"(paper W2025 Eq. 7.4) and type='event' (paper W2025 Eq. 7.6). "
                f"type={type!r} has no explicit paper closed-form cohort-share "
                f"weighting; use weights='cell' (default) for "
                f"jwdid_estat-style cell-count weighting."
            )

        gt = self.group_time_effects
        cell_weights = self._gt_weights
        n_g_per_cohort = self._n_g_per_cohort
        vcov = self._gt_vcov
        keys_ordered = self._gt_keys if self._gt_keys else sorted(gt.keys())

        # Map each cell to its un-normalized weight under the active scheme.
        # The aggregation step normalizes by ``w_total`` per aggregation
        # key, so only relative magnitudes matter here. For the cohort_share
        # scheme, the per-cell weight is ``N_g`` (paper Eqs. 7.4, 7.6)
        # — the same per-cell value across simple-overall and event-time;
        # the per-key normalization differs because the cell sets differ
        # (event-time aggregations group cells with the same ``k = t - g``,
        # so the denominator picks up only cohorts present at event-time
        # ``k`` per paper Eq. 7.6).
        def _cell_weight(c: Tuple[Any, Any]) -> float:
            if weights == "cell":
                return float(cell_weights.get(c, 0))
            # cohort_share
            return float(n_g_per_cohort.get(c[0], 0))

        def _agg_se(w_vec: np.ndarray) -> float:
            """Delta-method SE for a linear combination w'β given full vcov."""
            if vcov is None or len(w_vec) != vcov.shape[0]:
                return float("nan")
            return float(np.sqrt(max(w_vec @ vcov @ w_vec, 0.0)))

        # Compute BM contrast DOFs lazily for hc2_bm. ``cells_by_key`` is an
        # ordered mapping of aggregation_key -> list of (g, t) cells; the
        # contrast for each key sums the per-cell one-hot vectors weighted
        # by the active scheme's normalized per-cell weight. Returns a dict
        # mapping aggregation_key -> df (or NaN on fail-closed). For
        # non-hc2_bm, returns an empty dict (caller falls back to
        # ``self._df_survey``). Rebuilds the contrast column under the
        # active ``weights`` scheme so the BM DOF matches the actual SE
        # computation.
        def _bm_contrast_dofs_for(
            cells_by_key: Dict[Any, List[Tuple[Any, Any]]],
        ) -> Dict[Any, float]:
            if self.vcov_type != "hc2_bm" or self._bm_artifacts is None:
                return {}
            # ``X_red`` / ``bread_red`` are the REDUCED kept-column artifacts
            # from ``_fit_ols`` (post rank-deficient drops). ``coef_idx_map``
            # maps (g, t) → column index in ``X_red``. See
            # ``_bm_artifacts`` docstring above for the rationale.
            X_red, cluster_ids_full, bread_red, coef_idx_map = self._bm_artifacts
            n_red = X_red.shape[1]
            contrast_cols: List[np.ndarray] = []
            agg_keys: List[Any] = []
            for agg_key, cells in cells_by_key.items():
                if not cells:
                    continue
                w_total = sum(_cell_weight(c) for c in cells)
                if w_total == 0:
                    continue
                col = np.zeros(n_red)
                contributed = False
                for c in cells:
                    if c not in coef_idx_map:
                        continue
                    col[coef_idx_map[c]] = _cell_weight(c) / w_total
                    contributed = True
                if not contributed:
                    continue
                contrast_cols.append(col)
                agg_keys.append(agg_key)
            if not contrast_cols:
                return {k: float("nan") for k in cells_by_key}
            from diff_diff.linalg import _compute_cr2_bm_contrast_dof

            contrasts_matrix = np.column_stack(contrast_cols)
            dof_map: Dict[Any, float] = {}
            try:
                dof_vec = _compute_cr2_bm_contrast_dof(
                    X_red, cluster_ids_full, bread_red, contrasts_matrix
                )
                for i, k in enumerate(agg_keys):
                    candidate = float(dof_vec[i])
                    dof_map[k] = candidate if np.isfinite(candidate) else float("nan")
            except (ValueError, np.linalg.LinAlgError) as exc:
                warnings.warn(
                    f"WooldridgeDiDResults.aggregate({type!r}) could not "
                    f"compute Bell-McCaffrey contrast DOF "
                    f"({exc.__class__.__name__}: {exc}). "
                    "Affected aggregated inference (t_stat / p_value / "
                    "conf_int) will be NaN to preserve the hc2_bm contract.",
                    UserWarning,
                    stacklevel=3,
                )
                for k in agg_keys:
                    dof_map[k] = float("nan")
            # Fill non-computed keys with NaN to fail-closed.
            for k in cells_by_key:
                dof_map.setdefault(k, float("nan"))
            return dof_map

        def _build_effect(
            att: float, se: float, df_for_inference: Optional[float]
        ) -> Dict[str, Any]:
            """Build an effect dict using ``df_for_inference`` for the t-distribution.

            When ``self.vcov_type == "hc2_bm"``, ``df_for_inference`` should be
            the BM contrast DOF (NaN → fail-closed). For ``classical`` /
            ``hc2`` (one-way, no survey) the residual DOF ``self._df_one_way``
            is used so per-key inference matches R ``lm()`` /
            ``coef_test()`` t-distribution. For hc1 / surveyed paths,
            ``self._df_survey`` (None → normal-theory) is used.

            Under ``weights="cohort_share"`` (variable
            ``cohort_share_inference_fail_closed=True``), the inference
            fields (t-stat / p-value / conf-int) are nulled to NaN
            because the analytical SE is conditional-on-shares and
            understates unconditional uncertainty per paper W2025
            Section 7.5. The point estimate and conditional-on-shares
            SE are still returned for reference.
            """
            if cohort_share_inference_fail_closed:
                return {
                    "att": att,
                    "se": se,
                    "t_stat": float("nan"),
                    "p_value": float("nan"),
                    "conf_int": (float("nan"), float("nan")),
                }
            if self.vcov_type == "hc2_bm":
                if df_for_inference is None or not np.isfinite(df_for_inference):
                    return {
                        "att": att,
                        "se": se,
                        "t_stat": float("nan"),
                        "p_value": float("nan"),
                        "conf_int": (float("nan"), float("nan")),
                    }
                t_stat, p_value, conf_int = safe_inference(
                    att, se, alpha=self.alpha, df=df_for_inference
                )
            elif (
                self.vcov_type in ("classical", "hc2")
                and self._df_one_way is not None
                and np.isfinite(self._df_one_way)
            ):
                t_stat, p_value, conf_int = safe_inference(
                    att, se, alpha=self.alpha, df=self._df_one_way
                )
            else:
                t_stat, p_value, conf_int = safe_inference(
                    att, se, alpha=self.alpha, df=self._df_survey
                )
            return {
                "att": att,
                "se": se,
                "t_stat": t_stat,
                "p_value": p_value,
                "conf_int": conf_int,
            }

        # Cohort-share scheme requires populated _n_g_per_cohort; raise an
        # informative error rather than silently returning zero-weighted
        # NaN aggregates.
        if weights == "cohort_share" and not n_g_per_cohort:
            raise ValueError(
                "weights='cohort_share' requires per-cohort unit counts "
                "(_n_g_per_cohort) populated at fit time; this Results "
                "object has none. Re-fit with the current WooldridgeDiD "
                "version, or use weights='cell' (default) on legacy fits."
            )

        # Survey + cohort_share composition is not yet supported. Codex R3
        # P0 fix: ``_n_g_per_cohort`` is populated as raw ``unit.nunique()``
        # counts, so composing design-weighted ATT estimates (survey TSL)
        # with unweighted cohort shares targets a mixed estimand that is
        # not paper W2025 Section 7's design-population cohort-share form.
        # Design-consistent cohort totals (survey-weighted unit totals per
        # cohort) require additional plumbing — fail-closed for now,
        # tracked in DEFERRED.md follow-up.
        if weights == "cohort_share" and self.survey_metadata is not None:
            raise ValueError(
                "aggregate(weights='cohort_share') is not yet supported on "
                "survey-weighted fits (survey_design is not None): the "
                "cohort-share weights would compose design-weighted ATTs "
                "with unweighted cohort shares, targeting a mixed estimand "
                "that is not paper W2025 Section 7's design-population "
                "cohort-share form. Design-consistent cohort totals are "
                "deferred to a follow-up; use weights='cell' (default) "
                "on survey-weighted fits."
            )

        # Cohort-share variance conditional-on-shares disclaimer (paper
        # W2025 Section 7.5 / Eq. 7.4-7.6 discussion). The analytical SE
        # computed below treats the cohort-share weights ``ω̂_g`` /
        # ``ω̂_{ge}`` as fixed at their realized values, which means the
        # SE understates the unconditional sampling uncertainty from
        # estimating the shares themselves. Per `feedback_no_silent_failures`
        # and codex R2 P1 fix, fail-closed on the inference fields
        # (NaN out t-stat / p-value / conf-int) and emit a UserWarning
        # explaining the conditional-on-shares contract. The POINT
        # estimate ``att`` (paper Eq. 7.4 / 7.6 hand-calc form) and the
        # ``se`` (conditional-on-shares delta method) are still computed
        # and returned for reference, but the inferential machinery is
        # nulled out until proper APE/GMM-style aggregate inference is
        # derived (tracked in DEFERRED.md).
        cohort_share_inference_fail_closed = weights == "cohort_share"
        if cohort_share_inference_fail_closed:
            warnings.warn(
                "weights='cohort_share' aggregation: the analytical SE and "
                "inference (t-stat / p-value / conf-int) computed by "
                "WooldridgeDiDResults.aggregate(..., weights='cohort_share') "
                "treat the cohort-share weights ω̂_g / ω̂_{ge} as fixed; "
                "this conditional-on-shares variance understates the "
                "unconditional sampling uncertainty per paper W2025 "
                "Section 7.5. The library fail-closes the t-stat / p-value "
                "/ conf-int fields to NaN until proper APE/GMM-style "
                "aggregate inference is derived (tracked in DEFERRED.md). The "
                "POINT estimate and conditional-on-shares SE are computed "
                "and returned for reference; use weights='cell' (default) "
                "for the analytical aggregation with full inference.",
                UserWarning,
                stacklevel=2,
            )

        if type == "simple":
            # Bootstrap interaction guard: when ``_bootstrap_used`` was set
            # by ``_fit_ols`` (the multiplier bootstrap overrode the
            # analytical ``overall_*`` fields), the default
            # ``weights="cell"`` path is a no-op (preserves bootstrap
            # inference). The opt-in ``weights="cohort_share"`` path is not
            # bootstrapped — re-fit with ``n_bootstrap=0`` to use the
            # analytical cohort-share inference, or wait for the deferred
            # bootstrap-cohort-share follow-up (tracked in DEFERRED.md).
            if self._bootstrap_used:
                if weights == "cell":
                    return self
                raise ValueError(
                    "aggregate(type='simple', weights='cohort_share') is "
                    "not supported on bootstrapped fits "
                    "(n_bootstrap > 0): the multiplier bootstrap was run "
                    "on the cell-count-weighted overall ATT at fit time, "
                    "and the cohort-share aggregation has no matching "
                    "bootstrap variant yet. Re-fit with n_bootstrap=0 to "
                    "use cohort-share + analytical inference."
                )
            # Recompute overall ATT + SE under the active weighting scheme.
            # Under weights="cell" the result matches what fit() populated
            # at machine precision (re-derived from the same cell weights);
            # under weights="cohort_share" the overall ATT, SE, and BM
            # contrast DOF (under hc2_bm) are recomputed with cohort-share
            # per-cell weights per paper Eq. 7.4.
            cells_simple = [(g, t) for (g, t) in keys_ordered if g > 0 and t >= g]
            cells_by_simple: Dict[Any, List[Tuple[Any, Any]]] = {"simple": cells_simple}
            dofs = _bm_contrast_dofs_for(cells_by_simple)
            if cells_simple:
                w_total = sum(_cell_weight(c) for c in cells_simple)
                if w_total > 0:
                    att = sum(_cell_weight(c) * gt[c]["att"] for c in cells_simple) / w_total
                    w_vec = np.array(
                        [
                            _cell_weight(c) / w_total if c in cells_simple else 0.0
                            for c in keys_ordered
                        ]
                    )
                    se = _agg_se(w_vec)
                    eff = _build_effect(att, se, dofs.get("simple"))
                    self.overall_att = eff["att"]
                    self.overall_se = eff["se"]
                    self.overall_t_stat = eff["t_stat"]
                    self.overall_p_value = eff["p_value"]
                    self.overall_conf_int = eff["conf_int"]
                    # Atomic metadata mutation — only after successful write.
                    self.aggregation_weights["simple"] = weights

        elif type == "group":
            cells_by_g: Dict[Any, List[Tuple[Any, Any]]] = {}
            for g in self.groups:
                cells_by_g[g] = [(g2, t) for (g2, t) in keys_ordered if g2 == g and t >= g]
            dofs = _bm_contrast_dofs_for(cells_by_g)
            result: Dict[Any, Dict] = {}
            for g, cells in cells_by_g.items():
                if not cells:
                    continue
                w_total = sum(_cell_weight(c) for c in cells)
                if w_total == 0:
                    continue
                att = sum(_cell_weight(c) * gt[c]["att"] for c in cells) / w_total
                w_vec = np.array(
                    [_cell_weight(c) / w_total if c in cells else 0.0 for c in keys_ordered]
                )
                se = _agg_se(w_vec)
                result[g] = _build_effect(att, se, dofs.get(g))
            self.group_effects = result
            self.aggregation_weights["group"] = weights

        elif type == "calendar":
            cells_by_t: Dict[Any, List[Tuple[Any, Any]]] = {}
            for t in self.time_periods:
                cells_by_t[t] = [(g, t2) for (g, t2) in keys_ordered if t2 == t and t >= g]
            dofs = _bm_contrast_dofs_for(cells_by_t)
            result = {}
            for t, cells in cells_by_t.items():
                if not cells:
                    continue
                w_total = sum(_cell_weight(c) for c in cells)
                if w_total == 0:
                    continue
                att = sum(_cell_weight(c) * gt[c]["att"] for c in cells) / w_total
                w_vec = np.array(
                    [_cell_weight(c) / w_total if c in cells else 0.0 for c in keys_ordered]
                )
                se = _agg_se(w_vec)
                result[t] = _build_effect(att, se, dofs.get(t))
            self.calendar_effects = result
            self.aggregation_weights["calendar"] = weights

        elif type == "event":
            # Paper W2025 Eq. 7.6 cohort-share-by-exposure weighting is
            # defined for post-treatment exposure times (k >= 0) only;
            # pre-treatment lead effects use a separate Eq. 7.7
            # construction with ``nw_it`` weights that the library does
            # not yet expose. Under ``weights="cohort_share"`` we
            # restrict event aggregation to ``k >= 0`` to avoid
            # silently applying Eq. 7.6 weights to negative-lead cells
            # (codex R4 P1 fix). Under ``weights="cell"`` the full event
            # range is preserved for backward compatibility (pre-period
            # leads serve as placebos under OLS + never_treated).
            if weights == "cohort_share":
                eligible_pairs = [(g, t) for (g, t) in keys_ordered if t - g >= 0]
            else:
                eligible_pairs = list(keys_ordered)
            all_k = sorted({t - g for (g, t) in eligible_pairs})
            cells_by_k: Dict[int, List[Tuple[Any, Any]]] = {}
            for k in all_k:
                cells_by_k[k] = [(g, t) for (g, t) in eligible_pairs if t - g == k]
            dofs = _bm_contrast_dofs_for(cells_by_k)
            result = {}
            for k, cells in cells_by_k.items():
                if not cells:
                    continue
                w_total = sum(_cell_weight(c) for c in cells)
                if w_total == 0:
                    continue
                att = sum(_cell_weight(c) * gt[c]["att"] for c in cells) / w_total
                w_vec = np.array(
                    [_cell_weight(c) / w_total if c in cells else 0.0 for c in keys_ordered]
                )
                se = _agg_se(w_vec)
                result[k] = _build_effect(att, se, dofs.get(k))
            self.event_study_effects = result
            self.aggregation_weights["event"] = weights

        return self

    def summary(self, aggregation: str = "simple") -> str:
        """Print formatted summary table.

        Parameters
        ----------
        aggregation : which aggregation to display ("simple", "group", "calendar", "event")
        """
        lines = [
            "=" * 70,
            "    Wooldridge Extended Two-Way Fixed Effects (ETWFE) Results",
            "=" * 70,
            f"Method:          {self.method}",
            f"Control group:   {self.control_group}",
            f"Observations:    {self.n_obs}",
            f"Treated units:   {self.n_treated_units}",
            f"Control units:   {self.n_control_units}",
            f"Cohort trends:   {self.cohort_trends}",
            f"Aggregation w:   {self.aggregation_weights.get(aggregation, 'cell')}",
            "-" * 70,
        ]

        if self.survey_metadata is not None:
            from diff_diff.results import _format_survey_block

            lines.extend(_format_survey_block(self.survey_metadata, 70))
            lines.append("-" * 70)

        # Conley spatial-HAC variance label (rendered only on the conley path;
        # a full vcov-family label for all families is a separate follow-up).
        if self.vcov_type == "conley":
            from diff_diff.results import _format_vcov_label

            _vlabel = _format_vcov_label(
                self.vcov_type,
                cluster_name=self.cluster_name,
                n_clusters=self.n_clusters,
                n_obs=self.n_obs,
                conley_lag_cutoff=self.conley_lag_cutoff,
            )
            if _vlabel:
                lines.append(f"Std. errors:     {_vlabel}")
                lines.append("-" * 70)

        def _fmt_row(label: str, att: float, se: float, t: float, p: float, ci: Tuple) -> str:
            from diff_diff.results import _get_significance_stars

            stars = _get_significance_stars(p) if not np.isnan(p) else ""
            ci_lo = f"{ci[0]:.4f}" if not np.isnan(ci[0]) else "NaN"
            ci_hi = f"{ci[1]:.4f}" if not np.isnan(ci[1]) else "NaN"
            return (
                f"{label:<22} {att:>10.4f}  {se:>10.4f}  {t:>8.3f}  "
                f"{p:>8.4f}{stars}   [{ci_lo}, {ci_hi}]"
            )

        ci_pct = f"{(1 - self.alpha) * 100:.0f}%"
        header = (
            f"{'Parameter':<22} {'Estimate':>10}  {'Std. Err.':>10}  "
            f"{'t-stat':>8}  {'P>|t|':>8}   [{ci_pct} CI]"
        )
        lines.append(header)
        lines.append("-" * 70)

        if aggregation == "simple":
            lines.append(
                _fmt_row(
                    "ATT (simple)",
                    self.overall_att,
                    self.overall_se,
                    self.overall_t_stat,
                    self.overall_p_value,
                    self.overall_conf_int,
                )
            )
        elif aggregation == "group" and self.group_effects:
            for g, eff in sorted(self.group_effects.items()):
                lines.append(
                    _fmt_row(
                        f"ATT(g={g})",
                        eff["att"],
                        eff["se"],
                        eff["t_stat"],
                        eff["p_value"],
                        eff["conf_int"],
                    )
                )
        elif aggregation == "calendar" and self.calendar_effects:
            for t, eff in sorted(self.calendar_effects.items()):
                lines.append(
                    _fmt_row(
                        f"ATT(t={t})",
                        eff["att"],
                        eff["se"],
                        eff["t_stat"],
                        eff["p_value"],
                        eff["conf_int"],
                    )
                )
        elif aggregation == "event" and self.event_study_effects:
            for k, eff in sorted(self.event_study_effects.items()):
                if k < -self.anticipation:
                    suffix = " [pre]"
                elif k < 0:
                    suffix = " [antic]"
                else:
                    suffix = ""
                label = f"ATT(k={k})" + suffix
                lines.append(
                    _fmt_row(
                        label,
                        eff["att"],
                        eff["se"],
                        eff["t_stat"],
                        eff["p_value"],
                        eff["conf_int"],
                    )
                )
        else:
            lines.append(f"  (call .aggregate({aggregation!r}) first)")

        lines.append("=" * 70)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert headline results to a dictionary.

        Returns
        -------
        Dict[str, Any]
            Canonical inference row plus scalar metadata. Detailed
            group-time / aggregated tables are available via
            ``to_dataframe(aggregation=...)``.
        """
        result = {
            "att": self.att,
            "se": self.se,
            "t_stat": self.t_stat,
            "p_value": self.p_value,
            "conf_int_lower": self.overall_conf_int[0],
            "conf_int_upper": self.overall_conf_int[1],
            "method": self.method,
            "control_group": self.control_group,
            "n_obs": self.n_obs,
            "n_treated_units": self.n_treated_units,
            "n_control_units": self.n_control_units,
            "anticipation": self.anticipation,
            "alpha": self.alpha,
            "vcov_type": self.vcov_type,
        }
        if self.cluster_name is not None:
            result["cluster_name"] = self.cluster_name
        if self.n_clusters is not None:
            result["n_clusters"] = self.n_clusters
        if self.conley_lag_cutoff is not None:
            result["conley_lag_cutoff"] = self.conley_lag_cutoff
        return result

    def to_dataframe(self, aggregation: str = "event") -> pd.DataFrame:
        """Export aggregated effects to a DataFrame.

        Parameters
        ----------
        aggregation : "simple" | "group" | "calendar" | "event" | "gt"
            Use "gt" to export raw group-time effects.
        """
        if aggregation == "gt":
            rows = []
            for (g, t), eff in sorted(self.group_time_effects.items()):
                row = {"cohort": g, "time": t, "relative_period": t - g}
                row.update(eff)
                rows.append(row)
            return pd.DataFrame(rows)

        # Active weighting scheme for the requested aggregation surface
        # (default "cell"; flips to "cohort_share" after an opt-in
        # cohort-share aggregation on that surface). Stamped on every
        # exported row so downstream consumers can tell which estimand
        # the row represents without having to inspect the originating
        # Results object.
        active_weights = self.aggregation_weights.get(aggregation, "cell")
        mapping = {
            "simple": [
                {
                    "label": "ATT",
                    "att": self.overall_att,
                    "se": self.overall_se,
                    "t_stat": self.overall_t_stat,
                    "p_value": self.overall_p_value,
                    "conf_int_lo": self.overall_conf_int[0],
                    "conf_int_hi": self.overall_conf_int[1],
                    "cohort_trends": self.cohort_trends,
                    "aggregation_weights": active_weights,
                }
            ],
            "group": [
                {
                    "cohort": g,
                    **{k: v for k, v in eff.items() if k != "conf_int"},
                    "conf_int_lo": eff["conf_int"][0],
                    "conf_int_hi": eff["conf_int"][1],
                    "cohort_trends": self.cohort_trends,
                    "aggregation_weights": active_weights,
                }
                for g, eff in sorted((self.group_effects or {}).items())
            ],
            "calendar": [
                {
                    "time": t,
                    **{k: v for k, v in eff.items() if k != "conf_int"},
                    "conf_int_lo": eff["conf_int"][0],
                    "conf_int_hi": eff["conf_int"][1],
                    "cohort_trends": self.cohort_trends,
                    "aggregation_weights": active_weights,
                }
                for t, eff in sorted((self.calendar_effects or {}).items())
            ],
            "event": [
                {
                    "relative_period": k,
                    **{kk: vv for kk, vv in eff.items() if kk != "conf_int"},
                    "conf_int_lo": eff["conf_int"][0],
                    "conf_int_hi": eff["conf_int"][1],
                    "cohort_trends": self.cohort_trends,
                    "aggregation_weights": active_weights,
                }
                for k, eff in sorted((self.event_study_effects or {}).items())
            ],
        }
        rows = mapping.get(aggregation, [])
        return pd.DataFrame(rows)

    def plot_event_study(self, weights: str = "cell", **kwargs) -> None:
        """Event study plot. Always calls ``aggregate('event', weights=weights)``.

        Parameters
        ----------
        weights : "cell" | "cohort_share", default "cell"
            Aggregation weighting scheme threaded into the underlying
            ``aggregate("event", ...)`` call. ``"cohort_share"`` produces
            paper W2025 Eq. 7.6 cohort-share-by-exposure weights
            (post-treatment ``k >= 0`` only); inference fields are
            fail-closed to NaN per the Section 7.5 conditional-on-shares
            contract documented in REGISTRY, and the plot **suppresses
            error bars / CI bands** to honor the fail-closed contract
            (the conditional-on-shares SE would build a misleading
            normal-theory CI in the plotter).
        **kwargs
            Forwarded to ``diff_diff.visualization.plot_event_study``.

        Notes
        -----
        The wrapper unconditionally re-aggregates the event study under
        the requested ``weights`` scheme. This avoids the stale-cache
        hazard where a prior ``plot_event_study(weights="cohort_share")``
        call would leave the cached ``event_study_effects`` restricted
        to ``k >= 0`` (per the Eq. 7.6 scope), and a subsequent
        ``plot_event_study()`` (default ``weights="cell"``) call would
        silently reuse the cohort-share-keyed cache instead of restoring
        the full event range including pre-period placebo leads.
        """
        # Always re-aggregate under the requested weighting scheme. The
        # aggregate() method replaces ``event_study_effects`` in place
        # per the existing contract, so this is cheap and avoids
        # cohort_share→cell (or any cross-scheme) stale-cache bugs.
        self.aggregate("event", weights=weights)

        from diff_diff.visualization import plot_event_study

        effects = {k: v["att"] for k, v in (self.event_study_effects or {}).items()}
        if weights == "cohort_share":
            # Honor the fail-closed inference contract per paper Section
            # 7.5: the conditional-on-shares SE understates unconditional
            # uncertainty, so passing the finite SE into the plotter
            # would let it render a normal-theory CI that contradicts
            # the NaN inference fields the aggregate() helper produces.
            # Pass NaN SEs so the plotter suppresses error bars / CI
            # bands. Locked by ``test_plot_event_study_cohort_share_suppresses_error_bars``.
            se = {k: float("nan") for k in (self.event_study_effects or {})}
        else:
            se = {k: v["se"] for k, v in (self.event_study_effects or {}).items()}
        plot_event_study(effects=effects, se=se, alpha=self.alpha, **kwargs)

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
        n_gt = len(self.group_time_effects)
        att_str = f"{self.overall_att:.4f}" if not np.isnan(self.overall_att) else "NaN"
        se_str = f"{self.overall_se:.4f}" if not np.isnan(self.overall_se) else "NaN"
        p_str = f"{self.overall_p_value:.4f}" if not np.isnan(self.overall_p_value) else "NaN"
        # Surface the active simple aggregation scheme (the one that
        # produced the printed ``overall_*`` values) + cohort_trends
        # flag so repr is self-describing for downstream consumers.
        simple_weights = self.aggregation_weights.get("simple", "cell")
        return (
            f"WooldridgeDiDResults("
            f"ATT={att_str}, SE={se_str}, p={p_str}, "
            f"n_gt={n_gt}, method={self.method!r}, "
            f"cohort_trends={self.cohort_trends}, "
            f"aggregation_weights={simple_weights!r})"
        )
