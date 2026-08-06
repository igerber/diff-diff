"""
Gardner (2022) Two-Stage Difference-in-Differences Estimator.

Implements the two-stage DiD estimator from Gardner (2022), "Two-stage
differences in differences". The method:
1. Estimates unit + time fixed effects on untreated observations only
2. Residualizes ALL outcomes using estimated FEs
3. Regresses residualized outcomes on treatment indicators (Stage 2)

Inference uses the GMM sandwich variance estimator from Butts & Gardner
(2022) that correctly accounts for first-stage estimation uncertainty.

Point estimates are identical to ImputationDiD (Borusyak et al. 2024);
the key difference is the variance estimator (GMM sandwich vs. conservative).

References
----------
Gardner, J. (2022). Two-stage differences in differences.
    arXiv:2207.05943.
Butts, K. & Gardner, J. (2022). did2s: Two-Stage
    Difference-in-Differences. R Journal, 14(1), 162-173.
"""

import dataclasses
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import factorized as sparse_factorized

from diff_diff._base import BaseEstimator
from diff_diff._deprecation import NOT_SUPPLIED
from diff_diff.aggregation import AggregationKit
from diff_diff.conley import (
    ConleyMetric,
    _compute_conley_meat,
    _serial_bartlett_kernel_matrix,
    _validate_conley_kwargs,
    _validate_meat_psd,
)
from diff_diff.two_stage_aggregation import (  # noqa: F401 (compat re-exports)
    _SPARSE_DENSE_THRESHOLD,
    _lsmr_certified_normal_solve,
    _LSMRUnconvergedError,
    _TwoStageAggregationMixin,
)
from diff_diff.two_stage_bootstrap import TwoStageDiDBootstrapMixin
from diff_diff.two_stage_results import (
    TwoStageBootstrapResults,  # noqa: F401
    TwoStageDiDResults,
)  # noqa: F401 (re-export)
from diff_diff.utils import safe_inference

if TYPE_CHECKING:
    # Forward reference for the Wave E.1 survey-design path. Imported under
    # TYPE_CHECKING to keep the runtime cost zero and avoid any future
    # circular-import surprises with diff_diff.survey.
    from diff_diff.survey import ResolvedSurveyDesign, SurveyDesign


# =============================================================================
# Wave D — Gardner GMM-corrected meat for SpilloverDiD
# =============================================================================


def _compute_gmm_corrected_meat(
    *,
    X_1_sparse: sparse.csr_matrix,
    X_10_sparse: sparse.csr_matrix,
    eps_10: np.ndarray,
    X_2: np.ndarray,
    eps_2: np.ndarray,
    vcov_type: Literal["hc1", "conley", "cluster"],
    cluster_ids: Optional[np.ndarray] = None,
    conley_coords: Optional[np.ndarray] = None,
    conley_cutoff_km: Optional[float] = None,
    conley_metric: Optional[ConleyMetric] = None,
    conley_kernel: str = "bartlett",
    conley_time: Optional[np.ndarray] = None,
    conley_unit: Optional[np.ndarray] = None,
    conley_lag_cutoff: Optional[int] = None,
    survey_weights: Optional[np.ndarray] = None,
    resolved_survey: Optional["ResolvedSurveyDesign"] = None,
    score_pad_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Gardner (2022) GMM first-stage uncertainty correction — unified IF meat.

    Returns the (p_2, p_2) meat matrix ``Psi' K Psi`` where:

        psi_i  = gamma_hat' @ x_{10,i} * eps_{10,i} - x_{2,i} * eps_{2,i}
        Psi    = [psi_1; ...; psi_n]                    shape (n, p_2)
        K      = path-dependent kernel matrix
        meat   = Psi' @ K @ Psi                         shape (p_2, p_2)

    The caller wraps with the bread ``A_22^{-1} = (X_2' W X_2)^{-1}``:
    ``V = A_22^{-1} @ meat @ A_22^{-1}``.

    **Methodology synthesis (Wave D + Wave E.1):**

    *Wave D* (no reference software combines all three): Butts (2021) §3.1
    gives the IF construction for spillover-aware DiD; Gardner (2022) §4
    gives the two-stage GMM sandwich; Conley (1999) gives the spatial kernel.

    *Wave E.1* (additional composition for ``survey_design=``): Gerber
    (2026, arXiv:2605.04124) Proposition 1 — Binder Taylor Series
    Linearization for IF representations of smooth functionals; explicitly
    derived for TwoStageDiD in the paper's Appendix — composed with the
    Wave D GMM correction. The composition is mechanical: the Wave D Psi
    (with survey weights threaded through gamma_hat / eps / bread) is
    aggregated to PSU level and passed to the audited Binder TSL meat
    helper :func:`diff_diff.survey._compute_stratified_meat_from_psu_scores`.

    **Kernel dispatch (no survey):**

    - ``vcov_type="hc1"``: ``K = I_n``; ``meat = Psi' @ Psi`` with
      ``n / (n - p_2)`` finite-sample multiplier.
    - ``vcov_type="cluster"``: ``K_ij = 1{cluster_i = cluster_j}``;
      ``meat = S_cluster' @ S_cluster`` where ``S_cluster[g] = sum_{i in g} psi_i``,
      with ``G/(G-1) * (n-1)/(n - p_2)`` finite-sample multiplier.
    - ``vcov_type="conley"``: ``K_ij = K_space(d_ij/h) * 1{cluster_i = cluster_j}``
      (cross-sectional) or panel-block decomposed (``conley_time`` /
      ``conley_unit`` / ``conley_lag_cutoff`` set). No finite-sample
      multiplier — preserves the ``conleyreg`` / Wave B convention.

    **Kernel dispatch (Wave E.1 — ``resolved_survey is not None``):**

    - ``vcov_type="hc1"``: aggregate Psi to PSU totals; call
      ``_compute_stratified_meat_from_psu_scores`` with strata + FPC +
      ``lonely_psu`` handling. No HC1 finite-sample multiplier (Binder
      TSL has its own ``(1-f_h) * n_h/(n_h-1)`` correction).
    - ``vcov_type="cluster"``: ``cluster_ids`` IS the PSU (via upstream
      ``_inject_cluster_as_psu``); identical to the HC1+survey branch.
    - ``vcov_type="conley"`` with ``conley_lag_cutoff = 0`` (cross-sectional):
      Wave E.2 stratified-Conley sandwich on PSU totals via
      :func:`_compute_stratified_conley_meat`. Aggregates Psi to PSU
      totals + derives per-PSU centroids as the mean of per-obs
      ``conley_coords``; for each stratum applies the Conley kernel
      between PSU centroids scaled by ``(1 - f_h) * n_h/(n_h-1)``.
      Cross-stratum kernel weights are zero by sampling design.
    - ``vcov_type="conley"`` with ``conley_lag_cutoff > 0`` (panel-block,
      Wave E.2 follow-up): the orchestrator at
      :func:`_compute_stratified_conley_meat` adds a within-PSU serial
      Bartlett HAC term (Newey-West 1987 form) onto the Wave E.2 spatial
      sandwich — ``meat = meat_spatial + meat_serial`` with disjoint
      index sets (matches the no-survey panel-block decomposition at
      :func:`diff_diff.conley._compute_conley_meat`). Serial term uses
      per-period within-stratum centering (Binder TSL form) and
      panel-wide per-stratum FPC. Requires an **effective PSU** —
      either explicit ``survey_design.psu`` OR ``cluster=<col>``
      injected as the effective PSU per Wave E.1's
      ``_inject_cluster_as_psu`` routing. No-effective-PSU survey
      designs (weights-only / strata-only WITHOUT a cluster fallback)
      raise ``NotImplementedError`` upstream at ``SpilloverDiD.fit``
      post-resolution because the pseudo-PSU fallback would silently
      zero the serial sum.

    **`gamma_hat` solve** (mirror of `TwoStageDiD._compute_gmm_variance`
    pattern at `two_stage.py:1886-1917`): factorize ``X_10' W X_10`` via
    ``sparse_factorized`` (fast path); fall back to the certified sparse
    LSMR solve (:func:`_lsmr_certified_normal_solve`) with UserWarning when
    factorization fails. ``W`` is the diagonal of
    ``survey_weights`` when provided (Hájek-normalized, ``sum_i w_i = n``);
    identity otherwise. ``gamma_hat`` has shape ``(p_1, p_2)``.

    **Note (saturation — three distinct conditions, additive):**

    1. HC1 saturation (Wave D, pre-existing): ``n - p_2 <= 0`` → NaN meat
       + UserWarning. Fires on both no-survey and survey paths whenever
       the HC1 multiplier is invoked.
    2. CR1 saturation (Wave D, pre-existing): same gate, CR1 message.
    3. Survey-saturation (Wave E.1, NEW): when
       ``_compute_stratified_meat_from_psu_scores`` returns
       ``(_, var_computed=False, legit_zero=0)`` → NaN meat + UserWarning
       mentioning ``df_survey`` so callers can ``pytest.warns(match="df_survey")``.
       Departure from ``two_stage.py::_compute_gmm_variance`` (lines
       ~2003-2005) which currently NaN-fails SILENTLY; Wave E.1 surfaces
       the diagnostic per ``feedback_no_silent_failures``.

    Parameters
    ----------
    X_1_sparse : sparse.csr_matrix, shape (n, p_1)
        Full-sample FE design (drop-first-unit + drop-first-time
        identification).
    X_10_sparse : sparse.csr_matrix, shape (n, p_1)
        FE design with treated AND exposed rows zeroed. Same column space
        as X_1_sparse.
    eps_10 : np.ndarray, shape (n,)
        Stage-1 residual on Omega_0 rows; equal to y on ~Omega_0 rows
        (the X_{10,i} = 0 product collapses the IF contribution to just
        the stage-2 term).
    X_2 : np.ndarray, shape (n, p_2)
        Stage-2 design (treatment + ring columns for SpilloverDiD).
    eps_2 : np.ndarray, shape (n,)
        Stage-2 residual ``y_tilde - X_2 @ coef``.
    vcov_type : {"hc1", "conley", "cluster"}
        Kernel dispatch.
    cluster_ids : np.ndarray of shape (n,), optional
        Cluster identifiers. Required for ``vcov_type="cluster"``;
        used as the product-kernel cluster mask under ``vcov_type="conley"``
        when supplied. HC1 path passes ``None``. Under the survey path
        with ``vcov_type="cluster"``, ``cluster_ids`` is the PSU label
        (caller injects via ``_inject_cluster_as_psu``).
    conley_coords, conley_cutoff_km, conley_metric, conley_kernel,
    conley_time, conley_unit, conley_lag_cutoff
        Conley spatial-HAC kwargs. Required when ``vcov_type="conley"``.
        See :func:`diff_diff.conley._compute_conley_meat` for semantics.
    survey_weights : np.ndarray of shape (n,), optional
        Hájek-normalized survey weights (``sum_i w_i = n``). When provided,
        enters at the ``gamma_hat`` solve and per-obs Psi construction
        (eps weighting). When None, the no-survey Wave D path applies.
    resolved_survey : ResolvedSurveyDesign, optional
        Resolved survey design with PSU / strata / FPC arrays for the
        Wave E.1 Binder TSL meat. Required when ``survey_weights`` is
        supplied and stratified-cluster variance is requested. When None,
        the no-survey Wave D dispatch applies.
    score_pad_mask : np.ndarray of shape (n_full,), bool, optional
        Wave E.3 split-length contract for the survey path. When supplied,
        the FIT-SAMPLE inputs ``X_1_sparse`` / ``X_10_sparse`` / ``eps_10`` /
        ``X_2`` / ``eps_2`` / ``survey_weights`` (all length ``n_fit`` where
        ``n_fit == int(np.sum(score_pad_mask))``) are used to build
        ``gamma_hat`` and the fit-sample ``Psi``; the helper then zero-pads
        ``Psi`` to full panel length via
        ``Psi_padded[score_pad_mask] = Psi_fit`` AFTER construction but
        BEFORE kernel dispatch. The kernel-dispatch arrays — ``cluster_ids``,
        ``conley_coords``, ``conley_time``, ``conley_unit``,
        ``resolved_survey`` — must be at FULL length ``n_full`` so the meat
        helpers (Binder TSL / stratified-Conley / serial Bartlett) see the
        full-domain PSU / strata / centroid / time geometry. Default
        ``None`` keeps the historic single-length contract (all per-row
        inputs at the same length); this is what TwoStageDiD and the
        no-survey SpilloverDiD path use. Adopted in SpilloverDiD Wave E.3
        to mirror R ``survey::svyrecvar(subset())`` (Lumley 2010 §2.5) +
        the in-library precedents at ``imputation.py:2175-2183`` and
        ``prep.py:1401-1432``. Fit-sample construction is critical for the
        drop-first stage-1 FE column space stability — see the in-code
        comment block at the score-pad branch below.

    Returns
    -------
    meat : np.ndarray of shape (p_2, p_2)
        The IF outer-product meat, including any finite-sample multiplier.
        Caller wraps with the bread for the full vcov.
    """
    n, p_2 = X_2.shape

    # Validate Conley kwargs explicitly here. SpilloverDiD's Wave D path
    # bypasses solve_ols's vcov computation, so _validate_vcov_args /
    # _validate_conley_kwargs would not otherwise fire on this call.
    #
    # Wave E.3: when `score_pad_mask` is provided, the kernel-dispatch arrays
    # (conley_coords / conley_time / conley_unit / cluster_ids / resolved_survey)
    # are at FULL panel length while X / eps_* / X_*_sparse are at fit length.
    # Validate Conley shapes against the full length so the validator sees
    # consistent dimensions with the post-pad Psi the kernel actually consumes.
    n_for_conley = len(score_pad_mask) if score_pad_mask is not None else n
    if vcov_type == "conley":
        _validate_conley_kwargs(
            conley_coords,
            conley_cutoff_km,
            conley_metric,  # validator raises ValueError if None
            conley_kernel,
            n_for_conley,
            time=conley_time,
            unit=conley_unit,
            lag_cutoff=conley_lag_cutoff,
            cluster_ids=cluster_ids,
        )

    # Wave E.2 (this PR): conley × survey is now supported via the
    # stratified-Conley sandwich on PSU totals. Dispatch happens inside
    # the vcov_type == "conley" branch below (Wave E.1 already routed
    # hc1 / cluster + survey to the Binder TSL helper).

    # 1. gamma_hat = (X_10' W X_10)^{-1} (X_1' W X_2). Mirror the existing
    #    TwoStageDiD method at two_stage.py:1886-1917 — sparse_factorized
    #    fast path with certified sparse-LSMR fallback + UserWarning on singular.
    #    When survey_weights is provided, X_10/X_1 cross-products use W.
    if survey_weights is not None:
        XtX_10 = X_10_sparse.T @ X_10_sparse.multiply(survey_weights[:, None])
        Xt1_X2 = X_1_sparse.T @ (X_2 * survey_weights[:, None])
    else:
        XtX_10 = X_10_sparse.T @ X_10_sparse  # (p_1, p_1) sparse
        Xt1_X2 = X_1_sparse.T @ X_2  # (p_1, p_2) dense

    try:
        solve_XtX = sparse_factorized(XtX_10.tocsc())
        if Xt1_X2.ndim == 1:
            gamma_hat = solve_XtX(Xt1_X2).reshape(-1, 1)
        else:
            gamma_hat = np.column_stack([solve_XtX(Xt1_X2[:, j]) for j in range(Xt1_X2.shape[1])])
    except RuntimeError as exc:
        warnings.warn(
            "SpilloverDiD Wave D GMM sandwich: sparse factorization of "
            f"(X_10' X_10) failed ({type(exc).__name__}); falling back to "
            "sparse LSMR. This may indicate a rank-deficient or "
            "near-singular Stage 1 design and SE estimates may be less "
            "reliable.",
            UserWarning,
            stacklevel=2,
        )
        gamma_hat = _lsmr_certified_normal_solve(
            XtX_10.tocsc(), Xt1_X2, context="SpilloverDiD Wave-D GMM meat"
        )

    # 2. Psi = (X_10 @ gamma_hat) * eps_10[:, None] - X_2 * eps_2[:, None].
    #    Under Wave E.1 survey path, weights enter via element-wise eps
    #    multiplication (mirrors TwoStageDiD's `weighted_eps_10 = w * eps_10`
    #    pattern at two_stage.py:1922-1925); the additional w factor on
    #    each obs preserves Hájek scaling because the downstream
    #    `_compute_stratified_meat_from_psu_scores` and the caller-side
    #    `(X_2' W X_2)^{-1}` bread together yield the design-consistent
    #    variance per Gerber (2026) Proposition 1.
    Psi_stage1 = X_10_sparse @ gamma_hat  # (n, p_2) dense
    if survey_weights is not None:
        weighted_eps_10 = survey_weights * eps_10
        weighted_eps_2 = survey_weights * eps_2
        Psi = Psi_stage1 * weighted_eps_10[:, None] - X_2 * weighted_eps_2[:, None]
    else:
        Psi = Psi_stage1 * eps_10[:, None] - X_2 * eps_2[:, None]

    if not np.all(np.isfinite(Psi)):
        # Defensive: NaN in Psi would propagate silently through Psi.T @ Psi.
        # Surface as a warning + return NaN meat so the downstream
        # safe_inference path NaN-propagates per `feedback_no_silent_failures`.
        warnings.warn(
            "SpilloverDiD Wave D GMM sandwich: Psi matrix contains "
            "non-finite values. Returning NaN meat; downstream inference "
            "will NaN-propagate. This usually indicates rank-deficient "
            "stage-1 FE design or non-finite residuals upstream.",
            UserWarning,
            stacklevel=2,
        )
        return np.full((p_2, p_2), np.nan)

    # Wave E.3 zero-pad: when caller supplies `score_pad_mask`, expand the
    # fit-sample Psi to the full panel length by placing fit-sample rows at
    # `score_pad_mask == True` positions and zeros elsewhere. This preserves
    # the documented "zero-pad scores to full panel + retain full-design
    # resolved survey" pattern (R `survey::svyrecvar(subset())` form;
    # `imputation.py:2175-2183` and `prep.py:1401-1432` precedent) AFTER the
    # gamma_hat/Psi construction is done on the fit-sample inputs (which
    # keeps the stage-1 FE column space unchanged and the gamma_hat solve
    # full-rank). The downstream kernel-dispatch helpers see full-length Psi
    # and reuse the full-length cluster_ids / conley_* / resolved_survey
    # the caller passes. Excluded rows contribute exactly zero score, so
    # the meat is mathematically equivalent to "compute meat on fit-sample
    # Psi then place sums into the full-domain bookkeeping".
    if score_pad_mask is not None:
        n_full = len(score_pad_mask)
        if Psi.shape[0] != int(np.sum(score_pad_mask)):
            raise ValueError(
                "_compute_gmm_corrected_meat: score_pad_mask "
                f"length-mismatch (fit-sample Psi has {Psi.shape[0]} rows, "
                f"score_pad_mask has {int(np.sum(score_pad_mask))} True "
                f"entries out of {n_full})."
            )
        Psi_padded = np.zeros((n_full, p_2), dtype=np.float64)
        Psi_padded[score_pad_mask] = Psi
        Psi = Psi_padded
        n = n_full  # downstream finite-sample multipliers / kernel dispatch
        # see the full-length Psi (length matches cluster_ids / conley_*
        # which the caller passes at full length under the survey path).

    # 3. Kernel dispatch.
    #
    # Wave E.1 survey path (`resolved_survey is not None`) overrides the
    # Wave D HC1 / cluster branches with Binder TSL on PSU-aggregated Psi.
    # The Conley + survey combination (Wave E.2 / E.2 follow-up) is
    # handled by the `vcov_type == "conley"` branch further down, which
    # routes `resolved_survey is not None` fits to
    # `_compute_stratified_conley_meat` (panel-aware stratified-Conley
    # sandwich + optional within-PSU serial Bartlett HAC).
    if resolved_survey is not None and vcov_type in ("hc1", "cluster"):
        return _compute_binder_tsl_meat(
            Psi,
            resolved_survey=resolved_survey,
            cluster_ids=cluster_ids if vcov_type == "cluster" else None,
        )

    if vcov_type == "hc1":
        # K = I_n: meat = Psi' Psi with HC1 finite-sample multiplier.
        # Fail closed when n - p_2 <= 0 (saturated design — every degree
        # of freedom consumed by the stage-2 design): the multiplier
        # n / (n - p_2) is undefined, so NaN-propagate per
        # `feedback_no_silent_failures` rather than clamping the
        # denominator and emitting finite SE on an underdetermined fit.
        if n - p_2 <= 0:
            warnings.warn(
                "SpilloverDiD Wave D HC1 sandwich: saturated stage-2 design "
                f"(n_obs={n}, effective_rank={p_2}, n-p_2={n - p_2} <= 0). "
                "The HC1 finite-sample multiplier n/(n-p) is undefined. "
                "Returning NaN meat so downstream inference NaN-propagates.",
                UserWarning,
                stacklevel=2,
            )
            return np.full((p_2, p_2), np.nan)
        meat_unscaled = Psi.T @ Psi
        multiplier = n / (n - p_2)
        meat = multiplier * meat_unscaled
    elif vcov_type == "cluster":
        if cluster_ids is None:
            raise ValueError(
                "_compute_gmm_corrected_meat: vcov_type='cluster' requires "
                "cluster_ids; got None."
            )
        # K_ij = 1{cluster_i = cluster_j}: aggregate Psi per-cluster then
        # outer-product. S_cluster[g] = sum_{i in g} psi_i.
        unique_clusters, cluster_indices = np.unique(cluster_ids, return_inverse=True)
        G = len(unique_clusters)
        # Mirror linalg.py:1942 — reject G<2 so the CR1 finite-sample
        # multiplier G/(G-1) doesn't fabricate finite output on a degenerate
        # one-cluster sample.
        if G < 2:
            raise ValueError(f"Need at least 2 clusters for cluster-robust SEs, got {G}")
        # Fail closed on saturated design (n - p_2 <= 0). The CR1
        # multiplier (n-1)/(n-p) is undefined; emitting finite SE here
        # would be silently wrong.
        if n - p_2 <= 0:
            warnings.warn(
                "SpilloverDiD Wave D CR1 sandwich: saturated stage-2 design "
                f"(n_obs={n}, effective_rank={p_2}, n-p_2={n - p_2} <= 0). "
                "The CR1 finite-sample multiplier (n-1)/(n-p) is undefined. "
                "Returning NaN meat so downstream inference NaN-propagates.",
                UserWarning,
                stacklevel=2,
            )
            return np.full((p_2, p_2), np.nan)
        S_cluster = np.zeros((G, p_2))
        for j in range(p_2):
            np.add.at(S_cluster[:, j], cluster_indices, Psi[:, j])
        meat_unscaled = S_cluster.T @ S_cluster
        # CR1 finite-sample multiplier: G/(G-1) * (n-1)/(n-p_2). Standard
        # cluster-robust convention (Stata, R `sandwich::vcovCL(type='CR1')`).
        multiplier = (G / (G - 1)) * ((n - 1) / (n - p_2))
        meat = multiplier * meat_unscaled
    elif vcov_type == "conley":
        if conley_coords is None or conley_cutoff_km is None or conley_metric is None:
            raise ValueError(
                "_compute_gmm_corrected_meat: vcov_type='conley' requires "
                "conley_coords, conley_cutoff_km, and conley_metric."
            )
        if resolved_survey is not None:
            # Wave E.2: stratified-Conley sandwich on PSU totals.
            # Wave E.2 follow-up: with conley_lag_cutoff > 0 the orchestrator
            # adds the within-PSU serial Bartlett HAC term onto the spatial
            # sandwich (separable form; meat = meat_spatial + meat_serial).
            # cluster_ids is intentionally NOT threaded through — after PSU
            # aggregation every PSU is its own cluster, so a cluster product
            # kernel would zero all cross-PSU pairs. Wave E.1's
            # _resolve_effective_cluster path already coerced any
            # user-supplied cluster=<col> into PSU upstream.
            meat = _compute_stratified_conley_meat(
                Psi,
                conley_coords=np.asarray(conley_coords, dtype=np.float64),
                conley_cutoff_km=conley_cutoff_km,
                conley_metric=conley_metric,
                conley_kernel=conley_kernel,
                resolved_survey=resolved_survey,
                conley_time=conley_time,  # panel-aware per-period sandwich
                conley_lag_cutoff=conley_lag_cutoff,  # Wave E.2 follow-up serial term
            )
        else:
            # Wave D no-survey Conley path UNCHANGED — bit-identical fallback.
            # No finite-sample multiplier on the Conley path (matches conleyreg
            # / Wave B convention).
            meat = _compute_conley_meat(
                Psi,
                conley_coords,
                conley_cutoff_km,
                conley_metric,
                conley_kernel,
                time=conley_time,
                unit=conley_unit,
                lag_cutoff=conley_lag_cutoff,
                cluster_ids=cluster_ids,
            )
    else:
        raise ValueError(
            f"_compute_gmm_corrected_meat: vcov_type must be one of "
            f"'hc1', 'conley', 'cluster'; got {vcov_type!r}."
        )

    return meat


def _compute_binder_tsl_meat(
    Psi: np.ndarray,
    *,
    resolved_survey: "ResolvedSurveyDesign",
    cluster_ids: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Wave E.1 Binder TSL meat on PSU-aggregated Psi.

    Composes Gerber (2026, arXiv:2605.04124) Proposition 1 (Binder Taylor
    Series Linearization for IF representations of smooth functionals;
    explicitly derived for TwoStageDiD in the paper's Appendix) with the
    Wave D Gardner GMM Psi to produce a design-consistent variance for
    SpilloverDiD's stage-2 inference.

    The composition: aggregate per-obs Psi to PSU totals
    (``S_psu[g] = sum_{i in PSU g} Psi[i]``), then pass to
    :func:`diff_diff.survey._compute_stratified_meat_from_psu_scores` which
    applies the stratified-cluster formula
    ``meat = sum_h (1-f_h) * n_h/(n_h-1) * sum_j (S_hj - S_h_bar)(S_hj - S_h_bar)'``
    with FPC and ``lonely_psu`` handling.

    Mirrors ``TwoStageDiD._compute_gmm_variance`` at
    ``two_stage.py:1959-2005`` (the stratified-meat branch); the chief
    departure is that Wave E.1 surfaces a UserWarning on the saturated
    failure (``df_survey = 0``) rather than NaN-failing silently, per
    ``feedback_no_silent_failures``.

    Parameters
    ----------
    Psi : np.ndarray of shape (n, p_2)
        Per-obs influence-function scores (already weighted by survey
        weights via the upstream eps multiplication).
    resolved_survey : ResolvedSurveyDesign
        ``.psu`` may be None; when absent, each observation is treated as
        its own singleton PSU (matches the implicit-PSU convention of
        ``ResolvedSurveyDesign.df_survey`` no-PSU branches at
        ``survey.py:619-627``). ``.strata`` and ``.fpc`` are optional;
        absent strata synthesizes a single stratum (mirrors
        ``TwoStageDiD._compute_gmm_variance`` at L1976-1977).
    cluster_ids : np.ndarray of shape (n,), optional
        Ignored under the survey path — PSU is the cluster (upstream
        ``_inject_cluster_as_psu`` substitutes it for any user-supplied
        ``cluster=<col>``). The parameter is retained on the signature
        for symmetry with the non-survey dispatch; callers may pass None.

    Returns
    -------
    meat : np.ndarray of shape (p_2, p_2)
        The Binder TSL meat on PSU-aggregated scores.
    """
    # Local import keeps the module load cheap and avoids any circular-import
    # surprise with diff_diff.survey, which is the typical aggregation site
    # for SurveyDesign-touching helpers.
    from diff_diff.survey import _compute_stratified_meat_from_psu_scores

    del cluster_ids  # PSU substitutes upstream; kept on signature for symmetry

    p_2 = Psi.shape[1]
    n_obs = Psi.shape[0]
    # When PSU is absent, each obs is its own singleton PSU — matches the
    # `ResolvedSurveyDesign.df_survey` no-PSU branches (`n_obs - n_strata`
    # / `n_obs - 1`) and degenerates the Binder formula to obs-level
    # stratified-cluster meat. The caller-side cluster injection has
    # already handled the `cluster=<col>` case by populating
    # `resolved_survey.psu` from the cluster column; this fallback covers
    # the documented no-PSU SurveyDesign configurations (weights-only,
    # strata-only).
    if resolved_survey.psu is None:
        psu_arr: np.ndarray = np.arange(n_obs, dtype=np.int64)
    else:
        psu_arr = np.asarray(resolved_survey.psu)
    # Single np.unique call gives unique values, first-occurrence indices,
    # and the inverse map in one pass — O(n log n) total. The previous
    # impl used `np.where(psu_arr == c)[0][0]` per PSU, making the
    # PSU→strata / PSU→fpc mapping O(G·n); under the no-PSU survey path
    # (`psu_arr = arange(n_obs)`) G == n_obs and the mapping becomes
    # quadratic in n_obs. R14 codex P2 fix.
    unique_psus, first_idx, psu_indices = np.unique(psu_arr, return_index=True, return_inverse=True)
    G = len(unique_psus)

    # Aggregate Psi to PSU totals: S_psu[g, :] = sum_{i in PSU g} Psi[i, :].
    S_psu = np.zeros((G, p_2))
    for j in range(p_2):
        np.add.at(S_psu[:, j], psu_indices, Psi[:, j])

    # Map observation-level strata + fpc to PSU level via first-occurrence
    # indices (one fancy-index per array — O(G) — instead of O(G·n)).
    # Strata synthesizes a single stratum when absent (Binder formula
    # degenerates to one-stratum cluster-robust under unstratified FPC).
    if resolved_survey.strata is not None:
        psu_strata = np.asarray(resolved_survey.strata)[first_idx]
    else:
        psu_strata = np.zeros(G, dtype=int)

    psu_fpc: Optional[np.ndarray] = None
    if resolved_survey.fpc is not None:
        psu_fpc = np.asarray(resolved_survey.fpc, dtype=np.float64)[first_idx]

    # Unstratified single-PSU is variance-unidentified (matches
    # `_compute_stratified_psu_meat` convention at survey.py:1225 which
    # treats n_psu < 2 with no strata as NaN).
    if resolved_survey.strata is None and G < 2:
        warnings.warn(
            "SpilloverDiD Wave E.1 survey sandwich: df_survey is undefined "
            f"(single PSU, no strata; G={G}). Returning NaN meat so "
            "downstream inference NaN-propagates.",
            UserWarning,
            stacklevel=2,
        )
        return np.full((p_2, p_2), np.nan)

    meat, _variance_computed, _legit_zero = _compute_stratified_meat_from_psu_scores(
        psu_scores=S_psu,
        psu_strata=psu_strata,
        fpc_per_psu=psu_fpc,
        lonely_psu=resolved_survey.lonely_psu,
    )

    # Wave E.1 survey-saturated NaN-fail (NEW; departs from TwoStageDiD
    # silent NaN-VCV at two_stage.py:2003-2005 per `feedback_no_silent_failures`).
    if not _variance_computed and _legit_zero == 0:
        warnings.warn(
            "SpilloverDiD Wave E.1 survey sandwich: df_survey = 0 "
            "(all strata removed by lonely_psu='remove' on single-PSU "
            "strata; no PSU contributed to the meat). Returning NaN meat "
            "so downstream inference NaN-propagates.",
            UserWarning,
            stacklevel=2,
        )
        return np.full((p_2, p_2), np.nan)

    return meat


def _compute_stratified_conley_meat(
    Psi: np.ndarray,
    *,
    conley_coords: np.ndarray,
    conley_cutoff_km: float,
    conley_metric,
    conley_kernel: str,
    resolved_survey: "ResolvedSurveyDesign",
    conley_time: Optional[np.ndarray] = None,
    conley_lag_cutoff: Optional[int] = None,
) -> np.ndarray:
    """Wave E.2 panel-aware stratified-Conley meat on PSU-by-time scores.

    Composes Conley (1999) spatial-HAC with Gerber (2026, arXiv:2605.04124)
    Proposition 1 Binder TSL (the Wave E.1 foundation) and the Wave D
    Gardner GMM first-stage uncertainty correction (Butts 2021 ss3.1 +
    Gardner 2022 ss4) applied to SpilloverDiD's ring-indicator stage-2
    design. Wave E.2 follow-up extends to ``conley_lag_cutoff > 0`` by
    summing the within-PSU serial Bartlett HAC term (Newey-West 1987)
    onto the spatial sandwich: ``meat = meat_spatial + meat_serial`` with
    disjoint index sets, exactly matching the no-survey panel-block
    decomposition at :func:`diff_diff.conley._compute_conley_meat`. No
    reference software combines panel-block Conley + Binder TSL + Gardner
    GMM correction on a two-stage influence function.

    **Panel-aware composition (preserves the library's panel Conley
    contract):** for each period ``t``, aggregate per-obs Psi to PSU
    totals WITHIN that period (``S_psu_t[g] = sum_{i in PSU g, time t}
    Psi[i]``); derive each PSU's spatial centroid as the mean of
    per-observation ``conley_coords`` (panel-constant — PSU is a sampling
    unit with fixed location); apply the per-stratum Conley sandwich on
    ``S_psu_t`` via
    :func:`diff_diff.survey._compute_stratified_conley_meat_from_psu_scores`
    (Binder FPC factor ``(1 - f_h) * n_h/(n_h-1)``); sum across periods.
    Cross-period spatial pairs are excluded by construction, matching the
    library's existing ``conley_lag_cutoff = 0`` semantic ("within-period
    spatial only") at :func:`diff_diff.conley._compute_conley_meat`.
    Cross-stratum kernel weights are zero by sampling design (strata are
    exact independence partitions).

    Parameters
    ----------
    Psi : np.ndarray of shape (n, p_2)
        Per-obs Wave D Gardner GMM influence-function scores (already
        Hajek-weighted via the Wave E.1 upstream eps multiplication).
    conley_coords : np.ndarray of shape (n, 2)
        Per-observation lat/lon (or generic 2D coordinates). Already
        validated finite upstream at ``spillover.py:_validate_spillover_inputs``;
        no defensive finiteness check on derived PSU centroids.
    conley_cutoff_km : float
        Conley spatial-HAC bandwidth in km (haversine) or the
        coord units (euclidean / callable).
    conley_metric : ConleyMetric
        ``"haversine"`` / ``"euclidean"`` / callable, per
        :mod:`diff_diff.conley`.
    conley_kernel : str
        ``"bartlett"`` or ``"uniform"``.
    resolved_survey : ResolvedSurveyDesign
        ``.psu`` may be None; when absent, each observation is treated as
        its own singleton PSU (matches the implicit-PSU convention of
        :class:`ResolvedSurveyDesign` no-PSU branches). ``.strata`` and
        ``.fpc`` are optional; absent strata synthesize a single stratum.
    conley_time : np.ndarray of shape (n,), optional
        Per-observation period label. When None, all observations are
        treated as a single period (T = 1; the per-period loop reduces to
        one iteration on the full Psi, which is the cross-sectional
        Wave E.2 design). When provided (the standard SpilloverDiD case),
        the per-period loop preserves the within-period spatial semantic.
    conley_lag_cutoff : int, optional
        Bartlett serial-HAC bandwidth ``L`` in panel periods (Wave E.2
        follow-up). When None or 0, the spatial term is the entire meat
        (shipped Wave E.2 behaviour); when ``> 0``, the within-PSU serial
        Bartlett HAC :func:`_compute_stratified_serial_bartlett_meat` is
        added to the spatial term. ``L > 0`` requires ``conley_time`` set
        (with ``conley_time is None`` the panel reduces to T=1 and the
        serial helper short-circuits to zero meat).

    Returns
    -------
    meat : np.ndarray of shape (p_2, p_2)
        Wave E.2 panel-aware stratified-Conley meat
        (``sum_t meat_t`` where ``meat_t`` is the within-stratum Conley
        sandwich on the period-``t`` PSU totals).

    Notes
    -----
    ``cluster_ids`` is intentionally not accepted: after PSU aggregation
    every PSU is its own cluster, so threading a cluster product kernel
    into the inner :func:`_compute_stratified_conley_meat_from_psu_scores`
    would zero all cross-PSU pairs (``1{cluster_j == cluster_k}`` = 0 for
    j != k). The Wave E.1 ``_resolve_effective_cluster`` path already
    collapsed any user-supplied ``cluster=<col>`` into PSU upstream.

    NaN-fails (with ``UserWarning``) when the inner survey helper
    returns ``(False, 0)`` for every period — i.e. no stratum contributed
    variance and none was a legitimate zero across any period. Mirrors the
    Wave E.1 Binder TSL saturation behavior; departs from TwoStageDiD's
    silent NaN-VCV at ``two_stage.py:2003-2005`` per
    ``feedback_no_silent_failures``.

    Reductions:

    - ``T = 1`` (single period or ``conley_time is None``): single-pass
      stratified-Conley sandwich on the full PSU totals (the original
      cross-sectional Wave E.2 design). With ``conley_lag_cutoff > 0``
      the serial helper short-circuits to zero meat (no cross-period
      pairs possible).
    - ``H = 1`` stratum, ``FPC = inf``: spatial term reduces to ``sum_t``
      plain Conley sandwich on per-period PSU totals; serial term (if
      ``conley_lag_cutoff > 0``) reduces to plain Newey-West Bartlett HAC
      on PSU totals.
    - Bandwidth -> 0 (``K = I``): spatial reduces to ``sum_t`` per-period
      within-stratum HC sandwich on PSU totals (NOT Wave E.1 Binder,
      which is over time-collapsed PSU totals); serial term unchanged
      (separable form).
    - ``conley_lag_cutoff = 0`` or ``None``: bit-identical to shipped
      Wave E.2 (no serial helper call; spatial-only meat).
    """
    from diff_diff.survey import _compute_stratified_conley_meat_from_psu_scores

    p_2 = Psi.shape[1]
    n_obs = Psi.shape[0]
    coords_arr = np.asarray(conley_coords, dtype=np.float64)

    # No-PSU fallback: each obs is its own singleton PSU. Matches Wave E.1
    # Binder TSL convention at _compute_binder_tsl_meat L450-451.
    if resolved_survey.psu is None:
        psu_arr: np.ndarray = np.arange(n_obs, dtype=np.int64)
    else:
        psu_arr = np.asarray(resolved_survey.psu)
    strata_arr_full = (
        np.asarray(resolved_survey.strata) if resolved_survey.strata is not None else None
    )
    fpc_arr_full = (
        np.asarray(resolved_survey.fpc, dtype=np.float64)
        if resolved_survey.fpc is not None
        else None
    )

    # Panel-constant PSU centroids for explicit-PSU layouts (R4 P1 fix).
    # The Wave E.2 registry / api contract specifies
    # ``centroid_g = mean over i in PSU g of conley_coords[i]`` (panel-wide,
    # not per-period). For a PSU containing multiple units at different
    # coordinates with finite_mask dropping different members across
    # periods, per-period recomputation would silently shift the spatial
    # kernel weights — that would be a documented-contract violation.
    # Compute once on the full active sample so each period's helper call
    # sees the SAME centroid for the same PSU.
    #
    # For implicit-PSU (pseudo-PSU = obs index), every pseudo-PSU appears
    # in exactly one period, so the per-period slice naturally produces
    # the obs's own coordinate as that pseudo-PSU's centroid — no precompute
    # needed. The dictionary stays None on that branch.
    coord_dim = coords_arr.shape[1]
    psu_value_to_centroid: Optional[dict] = None
    if resolved_survey.psu is not None:
        unique_psus_full, _, psu_indices_full = np.unique(
            psu_arr, return_index=True, return_inverse=True
        )
        G_full = len(unique_psus_full)
        psu_coord_sums_full = np.zeros((G_full, coord_dim))
        for d in range(coord_dim):
            np.add.at(psu_coord_sums_full[:, d], psu_indices_full, coords_arr[:, d])
        psu_counts_full = np.bincount(psu_indices_full, minlength=G_full).astype(np.float64)
        psu_centroids_full = psu_coord_sums_full / psu_counts_full[:, None]
        psu_value_to_centroid = {unique_psus_full[g]: psu_centroids_full[g] for g in range(G_full)}

    # Per-period loop: preserves the library's "within-period spatial only"
    # contract for conley_lag_cutoff = 0. PSU set, centroids, strata, and
    # FPC are re-built from the ACTIVE rows in each period (not from the
    # full panel) so implicit-PSU layouts (`resolved_survey.psu is None`,
    # i.e. one pseudo-PSU per observation) don't drag off-period
    # zero-padded entries into the kernel via centering. For explicit-PSU
    # balanced-panel layouts the per-period centroids equal the
    # panel-constant centroids (obs coords are time-invariant), so this
    # re-indexing is bit-identical to the prior naive panel-wide PSU
    # mapping on that branch.
    if conley_time is None:
        # Treat all obs as one period (cross-sectional fallback).
        time_arr = np.zeros(n_obs, dtype=np.int64)
    else:
        time_arr = np.asarray(conley_time)
    unique_times = np.unique(time_arr)

    # Saturation guard for unstratified single-PSU on the FULL panel.
    # The per-period helper invocation will also NaN-fail when no period
    # contributes variance, but this front-door check matches Wave E.1's
    # ergonomic "df_survey is undefined" message for the panel-level
    # degenerate case.
    if strata_arr_full is None and len(np.unique(psu_arr)) < 2:
        G_total = len(np.unique(psu_arr))
        warnings.warn(
            "SpilloverDiD Wave E.2 stratified-Conley sandwich: df_survey is "
            f"undefined (single PSU, no strata; G={G_total}). Returning NaN "
            "meat so downstream inference NaN-propagates.",
            UserWarning,
            stacklevel=2,
        )
        return np.full((p_2, p_2), np.nan)

    meat = np.zeros((p_2, p_2))
    _variance_computed = False
    _legit_zero = 0
    for t in unique_times:
        period_mask = time_arr == t
        Psi_t = Psi[period_mask]
        psu_arr_t = psu_arr[period_mask]
        coords_arr_t = coords_arr[period_mask]
        unique_psus_t, first_idx_t, psu_indices_t = np.unique(
            psu_arr_t, return_index=True, return_inverse=True
        )
        G_t = len(unique_psus_t)

        # Per-period PSU totals.
        S_psu_t = np.zeros((G_t, p_2))
        for j in range(p_2):
            np.add.at(S_psu_t[:, j], psu_indices_t, Psi_t[:, j])

        # Per-period PSU centroids: panel-constant for explicit-PSU
        # (look up from the precomputed dict to match the documented
        # ``centroid_g = mean over i in PSU g of conley_coords[i]``
        # panel-wide contract); per-period mean for implicit-PSU
        # (pseudo-PSU = obs, each appears in exactly one period, so the
        # per-period mean IS the obs's own coord).
        if psu_value_to_centroid is not None:
            psu_centroids_t = np.array([psu_value_to_centroid[v] for v in unique_psus_t])
        else:
            psu_coord_sums_t = np.zeros((G_t, coord_dim))
            for d in range(coord_dim):
                np.add.at(psu_coord_sums_t[:, d], psu_indices_t, coords_arr_t[:, d])
            psu_counts_t = np.bincount(psu_indices_t, minlength=G_t).astype(np.float64)
            psu_centroids_t = psu_coord_sums_t / psu_counts_t[:, None]

        # Per-period strata + fpc.
        if strata_arr_full is not None:
            psu_strata_t = strata_arr_full[period_mask][first_idx_t]
        else:
            psu_strata_t = np.zeros(G_t, dtype=int)
        psu_fpc_t: Optional[np.ndarray] = None
        if fpc_arr_full is not None:
            psu_fpc_t = fpc_arr_full[period_mask][first_idx_t]

        # Stratified Conley sandwich for period t.
        meat_t, var_t, legit_zero_t = _compute_stratified_conley_meat_from_psu_scores(
            psu_scores=S_psu_t,
            psu_strata=psu_strata_t,
            psu_coords=psu_centroids_t,
            cutoff=conley_cutoff_km,
            metric=conley_metric,
            kernel=conley_kernel,
            fpc_per_psu=psu_fpc_t,
            lonely_psu=resolved_survey.lonely_psu,
        )
        meat += meat_t
        _variance_computed = _variance_computed or var_t
        _legit_zero += legit_zero_t

    # Wave E.2 follow-up: serial Bartlett HAC term for conley_lag_cutoff > 0.
    # Sums onto the spatial meat with disjoint index sets (separable form,
    # NOT Driscoll-Kraay 2D-HAC). Lag=0 / None short-circuits — the helper
    # itself returns zero-meat for L<=0 or T<=1 so this branch never erodes
    # the lag=0 bit-identity guarantee with shipped Wave E.2 (test (a)).
    # `cluster_ids` intentionally not threaded (same rationale as the spatial
    # term: post-PSU-aggregation each PSU is its own cluster, the within-PSU
    # serial loop already iterates exactly the right scope; threading a
    # cluster product kernel would be a no-op).
    if conley_lag_cutoff is not None and conley_lag_cutoff > 0:
        meat_serial, var_serial, legit_zero_serial = _compute_stratified_serial_bartlett_meat(
            Psi,
            psu_arr=psu_arr,
            time_arr=time_arr,
            strata_arr_full=strata_arr_full,
            fpc_arr_full=fpc_arr_full,
            conley_lag_cutoff=int(conley_lag_cutoff),
            lonely_psu=resolved_survey.lonely_psu,
        )
        meat = meat + meat_serial
        _variance_computed = _variance_computed or var_serial
        _legit_zero += legit_zero_serial

    # Wave E.2 survey-saturated NaN-fail per `feedback_no_silent_failures`.
    if not _variance_computed and _legit_zero == 0:
        warnings.warn(
            "SpilloverDiD Wave E.2 stratified-Conley sandwich: df_survey = 0 "
            "(all strata removed by lonely_psu='remove' on single-PSU "
            "strata; no PSU contributed to the meat). Returning NaN meat "
            "so downstream inference NaN-propagates.",
            UserWarning,
            stacklevel=2,
        )
        return np.full((p_2, p_2), np.nan)

    # Finite + PSD guards on the COMBINED survey meat (spatial + serial).
    # Shares ``_validate_meat_psd`` with :func:`diff_diff.conley._compute_conley_meat`
    # so the survey panel-block path has the same diagnostic surface as the
    # no-survey path. The radial 1-D Bartlett spatial kernel and the
    # Newey-West Bartlett serial kernel are both practitioner
    # specializations that are NOT formally PSD-guaranteed; adding two
    # non-PSD-guaranteed terms can produce a more indefinite combined
    # meat, so the check matters most on the panel-block path. CI codex
    # R1 P2 fix.
    # ``{eigval:.2e}`` is a literal placeholder for ``_validate_meat_psd``;
    # only ``{conley_kernel!r}`` is interpolated by the f-string here.
    _validate_meat_psd(
        meat,
        error_msg=(
            "SpilloverDiD Wave E.2 stratified-Conley meat contains non-finite "
            "values; check Psi for NaN/Inf upstream of the sandwich."
        ),
        warning_template=(
            f"SpilloverDiD Wave E.2 stratified-Conley meat with conley_kernel="
            f"{conley_kernel!r} has a materially negative eigenvalue "
            "({eigval:.2e}); the variance estimator is not guaranteed "
            "PSD on this design. Both supported kernels (radial bartlett and "
            "uniform spatial) plus the hardcoded serial Bartlett term are "
            "practitioner specializations of Conley 1999 / Newey-West 1987 "
            "and are not formally PSD-guaranteed; consider varying "
            "conley_cutoff_km / conley_lag_cutoff, or reviewing the design "
            "for collinearity / degenerate residual structure."
        ),
        stacklevel=3,
    )

    return meat


def _compute_stratified_serial_bartlett_meat(
    Psi: np.ndarray,
    *,
    psu_arr: np.ndarray,
    time_arr: np.ndarray,
    strata_arr_full: Optional[np.ndarray],
    fpc_arr_full: Optional[np.ndarray],
    conley_lag_cutoff: int,
    lonely_psu: str,
) -> Tuple[np.ndarray, bool, int]:
    """Wave E.2 follow-up: within-PSU serial Bartlett HAC meat over time on
    PSU-aggregated per-period scores.

    Composes Newey-West (1987) serial Bartlett HAC with Conley (1999)'s panel
    block-decomposition convention, Binder (1983) FPC, and Gerber (2026,
    arXiv:2605.04124) Proposition 1 Binder TSL on Wave D Gardner GMM IFs
    (Butts 2021 ss3.1 + Gardner 2022 ss4). Sibling helper to the Wave E.2
    spatial orchestrator :func:`_compute_stratified_conley_meat`; the two
    terms sum with disjoint index sets to form the panel-block meat
    ``meat = meat_spatial + meat_serial`` exactly matching the no-survey
    panel-block decomposition at :func:`diff_diff.conley._compute_conley_meat`.

    **Composition (separable form, NOT Driscoll-Kraay 2D-HAC):**

        meat_serial = sum_h FPC_h * sum_{g in stratum h}
                       sum_{|t-s| <= L, t != s, present[g, t] & present[g, s]}
                         (1 - |t-s|/(L+1)) * S_centered_t[g] @ S_centered_s[g].T

    where ``S_psu_t[g] = sum_{i in PSU g, time t} Psi[i]`` is the per-period
    PSU total, ``S_centered_t[g] = S_psu_t[g] - S_bar_h(g)_t`` is the
    per-period within-stratum centered score (Binder TSL form — matches the
    spatial helper's centering exactly), and ``|t-s|`` is computed on PANEL-
    WIDE dense time codes ``np.unique(time_arr, return_inverse=True)``
    (matches :func:`diff_diff.conley._compute_conley_meat` panel-block
    convention at conley.py:934-939; mirrors R ``conleyreg::time_dist``).
    Serial Bartlett kernel weights are hardcoded regardless of the spatial
    ``conley_kernel`` argument (also matches the conley.py reference).

    **FPC convention (panel-wide per-stratum)** — STANDALONE Newey-West
    composition on stratified clusters, NOT by analogy to the Binder spatial
    helper at :func:`_compute_binder_tsl_meat`. The serial sum aggregates
    within-PSU temporal correlation across all observed periods — it is a
    PANEL-level construct, not a period-level construct. The cluster set for
    the panel-level sum is the panel-wide set of PSUs in stratum h, so the
    FPC denominator uses ``n_h_panel = |unique PSUs in stratum h across the
    active sample|``, not the per-period ``n_h_t``. The spatial term keeps
    its per-period FPC unchanged (the period-t spatial sum IS a within-period
    stratified-cluster sandwich at one time index). For balanced panels with
    PSU present in every period, ``n_h_panel = n_h_t`` for all t so the two
    converge; the difference surfaces under unbalanced panels.

    **Centering asymmetry vs no-survey reference** — `conley.py:949-965`
    uses RAW scores for the serial term (no centering) because the no-survey
    path assumes ``E[scores] = 0`` under correct specification (X*eps
    centered around zero), so centering is a no-op. The survey-weighted
    Binder TSL form estimates the within-stratum mean and centers explicitly
    (textbook stratified-cluster sandwich; the per-period stratum mean enters
    via Binder's finite-population variance derivation). Using raw scores in
    the survey case would inflate variance by twice the squared per-period
    stratum mean and would NOT reduce to the cross-sectional Wave E.2 form
    at lag=0.

    **Singleton-adjust panel-wide mean asymmetry** — for ``lonely_psu="adjust"``
    on a singleton stratum, the serial helper centers against the panel-wide
    mean of per-period PSU totals (averaged over all (g, t) with
    ``present[g, t]``), NOT the per-period within-stratum mean used by the
    spatial helper. The scope difference reflects the serial term's panel-
    level nature: a singleton stratum at the panel level has no within-
    stratum cross-PSU variation to demean against, so the only meaningful
    centering target is the panel-wide PSU mean. The ``continue``-skip-FPC
    pattern matches the spatial helper at :func:`_compute_stratified_conley_meat_from_psu_scores`
    L2007-2017 to avoid divide-by-zero on ``n_h_panel = 1``.

    Parameters
    ----------
    Psi : np.ndarray of shape (n, p_2)
        Per-obs Wave D Gardner GMM IF scores (Hajek-weighted upstream via
        Wave E.1 eps multiplication). Bit-identical to the spatial path Psi.
    psu_arr : np.ndarray of shape (n,)
        Per-obs PSU identifier. ``resolved_survey.psu`` or pseudo-PSU =
        obs-index per the orchestrator's no-PSU fallback.
    time_arr : np.ndarray of shape (n,)
        Per-obs period label; normalized to dense codes 0..T-1 internally.
    strata_arr_full : np.ndarray of shape (n,) or None
        Per-obs stratum. None synthesizes a single stratum.
    fpc_arr_full : np.ndarray of shape (n,) or None
        Per-obs FPC. None disables FPC scaling on the serial term
        (`(1-f_h) = 1`); the n_h/(n_h-1) factor is still applied.
    conley_lag_cutoff : int
        Bartlett serial-HAC bandwidth `L` in panel periods. Must be >= 1
        (T=1 or L=0 returns zeros via short-circuit).
    lonely_psu : str
        ``"remove"`` / ``"certainty"`` / ``"adjust"`` — singleton-stratum
        handling, matches the Wave E.2 spatial helper exactly.

    Returns
    -------
    meat : np.ndarray of shape (p_2, p_2)
        Serial Bartlett HAC meat.
    variance_computed : bool
        Whether any actual variance was contributed.
    legitimate_zero_count : int
        Strata that legitimately contribute zero (lonely_psu="certainty").

    Notes
    -----
    Does NOT thread ``cluster_ids``: after PSU aggregation every PSU is its
    own cluster, so a cluster product kernel would zero all cross-PSU pairs
    (Wave E.2 dispatch-boundary rationale). Inherits Wave E.1
    `_resolve_effective_cluster` warn-and-coerce-to-PSU upstream.

    Does NOT receive ``psu_value_to_centroid``: the serial kernel operates
    on temporal lag only (no spatial component), so PSU centroids are
    irrelevant for this term. Asymmetric vs the spatial helper which needs
    centroids.

    T = 1 (single observed period) or ``conley_lag_cutoff <= 0`` short-
    circuits to zero meat with no variance reported — the degenerate panel-
    block path, NOT a saturation diagnostic.
    """
    p_2 = Psi.shape[1]
    L = int(conley_lag_cutoff)

    # T=1 short-circuit: no cross-period pairs are possible.
    unique_times, time_indices = np.unique(time_arr, return_inverse=True)
    T = len(unique_times)
    if T <= 1 or L <= 0:
        return np.zeros((p_2, p_2)), False, 0

    # PSU-stratum constancy is enforced upstream by `_validate_unit_constant_survey`
    # at survey.py:1008-1048 (PSU is a unit-level survey column; a PSU's
    # stratum assignment is constant across all observations by sampling-
    # design invariant). No defensive re-check here; the validator would have
    # raised before we reach the meat construction.

    # Build per-PSU per-period score tensor S_psu_panel[g, t, :].
    unique_psus, first_idx_panel, psu_indices_full = np.unique(
        psu_arr, return_index=True, return_inverse=True
    )
    G_panel = len(unique_psus)
    S_psu_panel = np.zeros((G_panel, T, p_2))
    for j in range(p_2):
        np.add.at(S_psu_panel[:, :, j], (psu_indices_full, time_indices), Psi[:, j])

    # Presence mask: True iff PSU g has at least one obs at period t.
    counts = np.zeros((G_panel, T), dtype=np.int64)
    np.add.at(counts, (psu_indices_full, time_indices), 1)
    present = counts > 0

    # Per-PSU panel-wide attributes (stratum + FPC).
    if strata_arr_full is not None:
        psu_strata_panel = np.asarray(strata_arr_full)[first_idx_panel]
    else:
        psu_strata_panel = np.zeros(G_panel, dtype=int)
    if fpc_arr_full is not None:
        psu_fpc_panel: Optional[np.ndarray] = np.asarray(fpc_arr_full, dtype=np.float64)[
            first_idx_panel
        ]
    else:
        psu_fpc_panel = None

    # Per-period within-stratum centering on the (G_panel, T, p_2) tensor.
    # Match the spatial helper's per-period stratum-mean centering exactly.
    # Initialize to ZEROS (NOT raw S_psu_panel) so any (g, t) cell whose
    # stratum-period has < 2 active PSUs contributes zero to downstream
    # serial cross-products. Leaving raw scores in singleton-active-period
    # cells would feed uncentered values into the serial Bartlett sum and
    # contaminate the covariance — codex R1 P1 fix. With < 2 active PSUs in
    # stratum h at period t, within-stratum variance is undefined; the
    # methodologically-correct behavior is zero contribution from that
    # (h, t) leg, matching the spatial helper's lonely_psu="remove"
    # convention applied at the per-period level.
    S_centered = np.zeros_like(S_psu_panel)
    unique_strata_panel = np.unique(psu_strata_panel)
    for t in range(T):
        for h in unique_strata_panel:
            active_mask = (psu_strata_panel == h) & present[:, t]
            if int(active_mask.sum()) < 2:
                # Singleton/empty active PSUs in stratum h at period t:
                # leave S_centered as zero (no contribution to serial sum
                # from this leg). The per-stratum singleton branch below
                # still handles panel-wide n_h_panel < 2.
                continue
            stratum_mean_t = S_psu_panel[active_mask, t, :].mean(axis=0)
            S_centered[active_mask, t, :] = S_psu_panel[active_mask, t, :] - stratum_mean_t

    # Panel-wide PSU mean for the singleton-adjust branch (compute lazily —
    # only needed if lonely_psu == "adjust" AND any singleton stratum exists).
    _global_psu_mean: Optional[np.ndarray] = None
    if lonely_psu == "adjust":
        present_count = int(present.sum())
        if present_count > 0:
            _global_psu_mean = (S_psu_panel * present[:, :, None]).sum(axis=(0, 1)) / present_count

    # Per-stratum serial accumulation.
    meat = np.zeros((p_2, p_2))
    _variance_computed = False
    legitimate_zero_count = 0
    t_codes_full = np.arange(T, dtype=np.float64)

    for h in unique_strata_panel:
        stratum_psus = np.where(psu_strata_panel == h)[0]
        n_h_panel = len(stratum_psus)

        # Singleton-stratum branch (mirror spatial helper at survey.py:2001-2017
        # — FPC `n_h/(n_h-1)` divides by zero when n_h_panel = 1 so MUST continue
        # to skip the multi-PSU FPC block below).
        if n_h_panel < 2:
            if lonely_psu == "remove":
                continue
            elif lonely_psu == "certainty":
                legitimate_zero_count += 1
                continue
            elif lonely_psu == "adjust":
                # Center against panel-wide PSU mean (different scope from
                # spatial helper's per-period stratum mean; see docstring
                # "singleton-adjust panel-wide mean asymmetry"). The guard
                # below covers the all-empty-presence edge (present_count = 0
                # at the global mean computation above leaves _global_psu_mean
                # None); in that pathological case every PSU's present mask
                # is all-False so the inner loop continues without subtraction.
                if _global_psu_mean is None:
                    continue
                for g in stratum_psus:
                    present_g = present[g]
                    if int(present_g.sum()) < 2:
                        continue
                    t_g = t_codes_full[present_g]
                    K_g = _serial_bartlett_kernel_matrix(t_g, L)
                    S_g_centered = S_psu_panel[g, present_g] - _global_psu_mean
                    with np.errstate(invalid="ignore", over="ignore"):
                        meat += S_g_centered.T @ K_g @ S_g_centered
                _variance_computed = True
                continue

        # Multi-PSU branch (n_h_panel >= 2): standard FPC + per-PSU serial.
        f_h_panel = 0.0
        if psu_fpc_panel is not None:
            N_h = psu_fpc_panel[stratum_psus[0]]
            if N_h < n_h_panel:
                raise ValueError(
                    f"FPC ({N_h}) is less than the number of PSUs "
                    f"({n_h_panel}) in stratum (Wave E.2 follow-up serial helper). "
                    "FPC must be >= n_PSU_panel."
                )
            f_h_panel = n_h_panel / N_h

        M_h_serial = np.zeros((p_2, p_2))
        for g in stratum_psus:
            present_g = present[g]
            if int(present_g.sum()) < 2:
                continue
            t_g = t_codes_full[present_g]
            # PANEL-WIDE dense time codes for the serial kernel (NOT per-PSU
            # positional encoding). See test (g) in TestSpilloverDiDWaveE2Followup
            # for the methodology lock; matches conley.py R-deviation.
            K_g = _serial_bartlett_kernel_matrix(t_g, L)
            S_g_centered = S_centered[g, present_g]
            with np.errstate(invalid="ignore", over="ignore"):
                M_h_serial += S_g_centered.T @ K_g @ S_g_centered

        fpc_scale = (1.0 - f_h_panel) * n_h_panel / (n_h_panel - 1)
        meat += fpc_scale * M_h_serial
        _variance_computed = True

    return meat, _variance_computed, legitimate_zero_count


# =============================================================================
# Main Estimator
# =============================================================================


class TwoStageDiD(TwoStageDiDBootstrapMixin, _TwoStageAggregationMixin, BaseEstimator):
    """
    Gardner (2022) two-stage Difference-in-Differences estimator.

    This estimator addresses TWFE bias under heterogeneous treatment
    effects by:
    1. Estimating unit + time FEs on untreated observations only
    2. Residualizing ALL outcomes using estimated FEs
    3. Regressing residualized outcomes on treatment indicators

    Point estimates are identical to ImputationDiD (Borusyak et al. 2024).
    The key difference is the variance estimator: TwoStageDiD uses a GMM
    sandwich variance that accounts for first-stage estimation uncertainty,
    while ImputationDiD uses the conservative variance from Theorem 3.

    Parameters
    ----------
    anticipation : int, default=0
        Number of periods before treatment where effects may occur.
    alpha : float, default=0.05
        Significance level for confidence intervals.
    cluster : str, optional
        Column name for cluster-robust standard errors.
        If None, clusters at the unit level by default.
    n_bootstrap : int, default=0
        Number of bootstrap iterations. If 0, uses analytical GMM
        sandwich inference.
    bootstrap_weights : str, default="rademacher"
        Type of bootstrap weights: "rademacher", "mammen", or "webb".
    seed : int, optional
        Random seed for reproducibility.
    rank_deficient_action : str, default="warn"
        Action when design matrix is rank-deficient:
        - "warn": Issue warning and drop linearly dependent columns
        - "error": Raise ValueError
        - "silent": Drop columns silently
    horizon_max : int, optional
        Maximum event-study horizon. If set, event study effects are only
        computed for abs(h) <= horizon_max.
    pretrends : bool, default=False
        If True, event study includes pre-treatment horizons for visual
        pre-trends assessment. Pre-period effects should be ~0 under
        parallel trends. Only affects event_study aggregation; overall
        ATT and group aggregation are unchanged.
    vcov_type : str, default="hc1"
        Variance estimator family. Permanently narrow to ``{"hc1"}`` — the
        Gardner (2022) two-stage GMM cluster-sandwich. Analytical-sandwich
        families ``{"classical", "hc2", "hc2_bm"}`` and ``"conley"`` are
        rejected at ``__init__`` / ``fit()`` because the GMM-corrected meat
        folds first-stage estimation uncertainty into the score, leaving no
        single hat matrix on which hat-matrix leverage or Bell-McCaffrey
        Satterthwaite DOF can be defined. Use ``cluster=<col>`` to select the
        cluster level; ``cluster=None`` (the default) clusters at the unit
        level, so the summary renders the unit-cluster CR1 label.

    Attributes
    ----------
    results_ : TwoStageDiDResults
        Estimation results after calling fit().
    is_fitted_ : bool
        Whether the model has been fitted.

    Examples
    --------
    Basic usage:

    >>> from diff_diff import TwoStageDiD, generate_staggered_data
    >>> data = generate_staggered_data(n_units=200, seed=42)
    >>> est = TwoStageDiD()
    >>> results = est.fit(data, outcome='outcome', unit='unit',
    ...                   time='period', first_treat='first_treat')
    >>> results.print_summary()

    With a post-fit event study (M-022):

    >>> est = TwoStageDiD()
    >>> results = est.fit(data, outcome='outcome', unit='unit',
    ...                   time='period', first_treat='first_treat')
    >>> es = results.aggregate('event_study')
    >>> from diff_diff import plot_event_study
    >>> plot_event_study(es)

    Notes
    -----
    The two-stage estimator uses ALL untreated observations (never-treated +
    not-yet-treated periods of eventually-treated units) to estimate the
    counterfactual model.

    References
    ----------
    Gardner, J. (2022). Two-stage differences in differences.
        arXiv:2207.05943.
    Butts, K. & Gardner, J. (2022). did2s: Two-Stage
        Difference-in-Differences. R Journal, 14(1), 162-173.
    """

    def __init__(
        self,
        anticipation: int = 0,
        alpha: float = 0.05,
        cluster: Optional[str] = None,
        n_bootstrap: int = 0,
        bootstrap_weights: str = "rademacher",
        seed: Optional[int] = None,
        rank_deficient_action: str = "warn",
        horizon_max: Optional[int] = None,
        pretrends: bool = False,
        vcov_type: str = "hc1",
    ):
        if rank_deficient_action not in ("warn", "error", "silent"):
            raise ValueError(
                f"rank_deficient_action must be 'warn', 'error', or 'silent', "
                f"got '{rank_deficient_action}'"
            )
        if bootstrap_weights not in ("rademacher", "mammen", "webb"):
            raise ValueError(
                f"bootstrap_weights must be 'rademacher', 'mammen', or 'webb', "
                f"got '{bootstrap_weights}'"
            )
        self._validate_vcov_type(vcov_type)

        self.anticipation = anticipation
        self.alpha = alpha
        self.cluster = cluster
        self.vcov_type = vcov_type
        self.n_bootstrap = n_bootstrap
        self.bootstrap_weights = bootstrap_weights
        self.seed = seed
        self.rank_deficient_action = rank_deficient_action
        self.horizon_max = horizon_max
        self.pretrends = pretrends

        self.is_fitted_ = False
        self.results_: Optional[TwoStageDiDResults] = None

    def fit(
        self,
        data: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        first_treat: str,
        covariates: Optional[List[str]] = None,
        aggregate: Any = NOT_SUPPLIED,
        balance_e: Any = NOT_SUPPLIED,
        survey_design: Optional["SurveyDesign"] = None,
    ) -> TwoStageDiDResults:
        """
        Fit the two-stage DiD estimator.

        Parameters
        ----------
        data : pd.DataFrame
            Panel data with unit and time identifiers.
        outcome : str
            Name of outcome variable column.
        unit : str
            Name of unit identifier column.
        time : str
            Name of time period column.
        first_treat : str
            Name of column indicating when unit was first treated.
            Use 0 (or np.inf) for never-treated units.
        covariates : list of str, optional
            List of covariate column names.
        aggregate : str, optional
            DEPRECATED (3.9, removed in 4.0; row M-022): aggregate as a
            post-fit step instead — ``results.aggregate('event_study')`` /
            ``.aggregate('group')`` / ``.aggregate('simple')``. Supplying
            ANY value (``None`` included) warns; the deprecated path still
            works and returns exactly the numbers it always did
            (fit-time mode: None/"simple" overall only, "event_study",
            "group", or "all").
        balance_e : int, optional
            DEPRECATED (3.9, removed in 4.0; row M-119): moves onto
            ``results.aggregate('event_study', balance_e=...)``. Restricts
            the event study to cohorts observed at every relative time in
            ``[-balance_e, max_h]`` (the balanced-window rule).
        survey_design : SurveyDesign, optional
            Survey design specification for design-based inference. Supports
            pweight only (aweight/fweight raise ValueError). Supports strata,
            PSU, and FPC for design-based GMM sandwich variance. Strata enters
            survey df for t-distribution inference.
            Both analytical (n_bootstrap=0) and bootstrap inference are supported.

        Returns
        -------
        TwoStageDiDResults
            Object containing all estimation results.

        Raises
        ------
        ValueError
            If required columns are missing or data validation fails.
        """
        # M-022/M-119 deprecation shim (CS-style joint warning): a plain
        # fit() never warns; supplying EITHER param with ANY value (None
        # included) warns once, then the legacy routing below runs
        # unchanged - the deprecated path returns exactly the numbers it
        # always did (no new value validation; unknown strings still act
        # like None). The post-fit successor validates its own vocabulary.
        _deprecated_passed = [
            n
            for n, v in (("aggregate", aggregate), ("balance_e", balance_e))
            if v is not NOT_SUPPLIED
        ]
        if _deprecated_passed:
            _args = " / ".join(f"{n}=" for n in _deprecated_passed)
            warnings.warn(
                f"TwoStageDiD.fit({_args}) is deprecated and will be "
                "removed in 4.0. Fit once, then aggregate as a post-fit "
                "step: results = TwoStageDiD().fit(...); "
                "results.aggregate('event_study') / .aggregate('group') / "
                ".aggregate('simple'). balance_e moves onto aggregate() "
                "alongside it: results.aggregate('event_study', "
                "balance_e=2).",
                FutureWarning,
                stacklevel=2,
            )
        if aggregate is NOT_SUPPLIED:
            aggregate = None
        if balance_e is NOT_SUPPLIED:
            balance_e = None

        # Re-validate vcov_type at fit-time: set_params validates eagerly
        # (BaseEstimator probe re-init), so this only catches DIRECT
        # attribute mutation (est.vcov_type = ...).
        self._validate_vcov_type(self.vcov_type)

        # ---- Data validation ----
        required_cols = [outcome, unit, time, first_treat]
        if covariates:
            required_cols.extend(covariates)

        missing = [c for c in required_cols if c not in data.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # Create working copy
        df = data.copy()

        # Resolve survey design if provided
        from diff_diff.survey import (
            _inject_cluster_as_psu,
            _resolve_effective_cluster,
            _resolve_survey_for_fit,
            _validate_unit_constant_survey,
        )

        resolved_survey, survey_weights, survey_weight_type, survey_metadata = (
            _resolve_survey_for_fit(survey_design, data, "analytical")
        )

        _uses_replicate_ts = resolved_survey is not None and resolved_survey.uses_replicate_variance
        if _uses_replicate_ts and self.n_bootstrap > 0:
            raise ValueError(
                "Cannot use n_bootstrap > 0 with replicate-weight survey designs. "
                "Replicate weights provide their own variance estimation."
            )
        if _uses_replicate_ts and self.cluster is not None:
            raise NotImplementedError(
                "TwoStageDiD(cluster=...) with a replicate-weight survey design "
                "is not supported: replicate-weight variance "
                "(compute_replicate_refit_variance) estimates the SE by "
                "per-replicate re-fit and ignores cluster= entirely, so the "
                "cluster specification would be silently dropped. Use cluster= "
                "with analytical/TSL inference (no replicate weights), or a "
                "replicate-weight design without cluster=."
            )
        # Validate within-unit constancy for panel survey designs
        if resolved_survey is not None:
            _validate_unit_constant_survey(data, unit, survey_design)
            if resolved_survey.weight_type != "pweight":
                raise ValueError(
                    f"TwoStageDiD survey support requires weight_type='pweight', "
                    f"got '{resolved_survey.weight_type}'. The survey variance math "
                    f"assumes probability weights (pweight)."
                )
            # FPC is supported — threaded through _compute_stratified_meat_from_psu_scores()
            # in _compute_gmm_variance().

        # Bootstrap + survey supported via PSU-level multiplier bootstrap.

        df[time] = pd.to_numeric(df[time])
        df[first_treat] = pd.to_numeric(df[first_treat])

        # Validate absorbing treatment
        ft_nunique = df.groupby(unit)[first_treat].nunique()
        non_constant = ft_nunique[ft_nunique > 1]
        if len(non_constant) > 0:
            example_unit = non_constant.index[0]
            example_vals = sorted(df.loc[df[unit] == example_unit, first_treat].unique())
            warnings.warn(
                f"{len(non_constant)} unit(s) have non-constant '{first_treat}' "
                f"values (e.g., unit '{example_unit}' has values {example_vals}). "
                f"TwoStageDiD assumes treatment is an absorbing state "
                f"(once treated, always treated) with a single treatment onset "
                f"time per unit. Non-constant first_treat violates this assumption "
                f"and may produce unreliable estimates.",
                UserWarning,
                stacklevel=2,
            )
            df[first_treat] = df.groupby(unit)[first_treat].transform("first")

        # Identify treatment status
        df["_never_treated"] = (df[first_treat] == 0) | (df[first_treat] == np.inf)

        # Check for always-treated units
        min_time = df[time].min()
        always_treated_mask = (~df["_never_treated"]) & (df[first_treat] <= min_time)
        always_treated_units = df.loc[always_treated_mask, unit].unique()
        n_always_treated = len(always_treated_units)
        # `keep_mask` is always defined (defaults to all-True over `data`) so
        # downstream survey-path code can subset full-domain arrays uniformly
        # whether or not the always-treated branch fires (Wave E.3 parity).
        keep_mask = pd.Series(np.ones(len(data), dtype=bool), index=data.index)
        if n_always_treated > 0:
            unit_list = ", ".join(str(u) for u in always_treated_units[:10])
            suffix = f" (and {n_always_treated - 10} more)" if n_always_treated > 10 else ""
            survey_note = ""
            if survey_weights is not None or resolved_survey is not None:
                # Wave E.3 parity (PR #482 SpilloverDiD precedent): under the
                # always-treated drop we subset `survey_weights` for stage-1 /
                # stage-2 OLS arithmetic but retain the full-domain
                # `resolved_survey` for variance estimation. The Binder TSL
                # meat consumes zero-padded per-cluster scores keyed against
                # the full-domain PSU/strata layout, matching R
                # `survey::svyrecvar(subset())` (Lumley 2010 §2.5) and the
                # in-library convention used at `imputation.py:2175-2183`
                # (PreTrendsImputation) and `prep.py:1401-1432` (DCDH cell
                # variance).
                survey_note = (
                    " Associated survey weights subsetted for stage-1 / "
                    "stage-2 OLS; full-domain survey design retained for "
                    "variance estimation (Wave E.3 parity)."
                )
            warnings.warn(
                f"{n_always_treated} unit(s) are treated in all observed periods "
                f"(first_treat <= {min_time}): [{unit_list}{suffix}]. "
                "These units have no untreated observations and cannot contribute "
                f"to the counterfactual model. Excluding from estimation.{survey_note}",
                UserWarning,
                stacklevel=2,
            )
            df = df[~df[unit].isin(always_treated_units)].copy()

            # Stage-1 / stage-2 OLS sample = post-drop `df`; survey_weights
            # must be aligned for OLS arithmetic. The active-sample mask
            # `keep_mask` is FULL-DOMAIN length and identifies kept rows.
            keep_mask = ~data[unit].isin(always_treated_units)
            if survey_weights is not None:
                survey_weights = survey_weights[keep_mask.values]
            # NOTE (Wave E.3 parity): `resolved_survey` is intentionally NOT
            # subsetted here. Full-domain `n_psu` / `n_strata` / `df_survey` /
            # `strata` / `fpc` / `psu` arrays propagate downstream; the
            # always-treated rows contribute zero score to the Binder TSL
            # meat (see `_compute_gmm_variance` for the zero-pad mechanics).
            # `survey_metadata` is computed once on the full-domain design by
            # `_resolve_survey_for_fit` upstream and (when cluster injection
            # fires) recomputed once post-injection in the block below; no
            # post-always-treated recompute is needed.

        # Treatment indicator with anticipation
        effective_treat = df[first_treat] - self.anticipation
        df["_treated"] = (~df["_never_treated"]) & (df[time] >= effective_treat)

        # Partition into Omega_0 (untreated) and Omega_1 (treated)
        omega_0_mask = ~df["_treated"]
        omega_1_mask = df["_treated"]

        n_omega_0 = int(omega_0_mask.sum())
        n_omega_1 = int(omega_1_mask.sum())

        if n_omega_0 == 0:
            raise ValueError(
                "No untreated observations found. Cannot estimate counterfactual model."
            )
        if n_omega_1 == 0:
            raise ValueError("No treated observations found. Nothing to estimate.")

        # Groups and time periods
        time_periods = sorted(df[time].unique())
        treatment_groups = sorted([g for g in df[first_treat].unique() if g > 0 and g != np.inf])

        if len(treatment_groups) == 0:
            raise ValueError("No treated units found. Check 'first_treat' column.")

        # Unit info
        unit_info = (
            df.groupby(unit).agg({first_treat: "first", "_never_treated": "first"}).reset_index()
        )
        n_treated_units = int((~unit_info["_never_treated"]).sum())
        units_in_omega_0 = df.loc[omega_0_mask, unit].unique()
        n_control_units = len(units_in_omega_0)

        # Cluster variable
        cluster_var = self.cluster if self.cluster is not None else unit
        if self.cluster is not None and self.cluster not in df.columns:
            raise ValueError(
                f"Cluster column '{self.cluster}' not found in data. "
                f"Available columns: {list(df.columns)}"
            )

        # Resolve effective cluster and inject cluster-as-PSU for survey variance.
        # Wave E.3 parity: under always-treated drop, `resolved_survey` retains
        # full-domain arrays — so the cluster_ids fed into
        # `_resolve_effective_cluster` / `_inject_cluster_as_psu` MUST be the
        # FULL-DOMAIN cluster column (sourced from `data`, not `df` which is
        # post-drop). Otherwise the zip in `_inject_cluster_as_psu` between
        # `resolved.strata` (full-domain) and `cluster_ids` (post-drop) would
        # truncate silently. Mirrors `spillover.py:2879` (PR #482 Wave E.3
        # `cluster_ids_full` invariant).
        cluster_ids_full: Optional[np.ndarray] = None
        if resolved_survey is not None:
            cluster_ids_raw = (
                np.asarray(data[cluster_var].values) if cluster_var in data.columns else None
            )
            effective_cluster_ids = _resolve_effective_cluster(
                resolved_survey,
                cluster_ids_raw,
                cluster_var if self.cluster is not None else None,
            )
            resolved_survey = _inject_cluster_as_psu(resolved_survey, effective_cluster_ids)
            # When survey PSU is present, use it as the effective cluster for
            # GMM variance (PSU overrides unit-level clustering). Both
            # `df["_survey_cluster"]` (post-drop) and `cluster_ids_full`
            # (full-domain) reference the post-injection PSU labels: the OLS
            # aggregation downstream uses `df["_survey_cluster"]` (df-aligned);
            # `_compute_gmm_variance` receives `cluster_ids_full` (full-domain)
            # so the zero-padded per-PSU meat sees the full-domain PSU list.
            if resolved_survey.psu is not None:
                df["_survey_cluster"] = resolved_survey.psu[keep_mask.values]
                cluster_var = "_survey_cluster"
                cluster_ids_full = np.asarray(resolved_survey.psu)
            elif effective_cluster_ids is not None:
                # No PSU injected (e.g., user-only cluster=, no survey.psu);
                # full-domain cluster_ids match effective_cluster_ids.
                cluster_ids_full = np.asarray(effective_cluster_ids)
            # Recompute metadata after PSU injection. `raw_w` is taken from the
            # FULL-DOMAIN `data` so the design-effect / Kish-effective-n
            # diagnostics reflect the full domain (matches the retained
            # full-domain `resolved_survey` arrays per Wave E.3 R6 lesson).
            if resolved_survey.psu is not None and survey_metadata is not None:
                from diff_diff.survey import compute_survey_metadata

                # resolved_survey non-None implies survey_design was passed.
                assert survey_design is not None
                raw_w = (
                    np.asarray(data[survey_design.weights].values, dtype=np.float64)
                    if survey_design.weights
                    else np.ones(len(data), dtype=np.float64)
                )
                survey_metadata = compute_survey_metadata(resolved_survey, raw_w)

        # Relative time
        df["_rel_time"] = np.where(
            ~df["_never_treated"],
            df[time] - df[first_treat],
            np.nan,
        )

        # ---- Stage 1: OLS on untreated observations ----
        unit_fe, time_fe, grand_mean, delta_hat, kept_cov_mask = self._fit_untreated_model(
            df, outcome, unit, time, covariates, omega_0_mask, weights=survey_weights
        )

        # ---- Rank condition checks ----
        treated_unit_ids = df.loc[omega_1_mask, unit].unique()
        units_with_fe = set(unit_fe.keys())
        units_missing_fe = set(treated_unit_ids) - units_with_fe

        post_period_ids = df.loc[omega_1_mask, time].unique()
        periods_with_fe = set(time_fe.keys())
        periods_missing_fe = set(post_period_ids) - periods_with_fe

        if units_missing_fe or periods_missing_fe:
            parts = []
            if units_missing_fe:
                sorted_missing = sorted(units_missing_fe)
                parts.append(
                    f"{len(units_missing_fe)} treated unit(s) have no untreated "
                    f"periods (units: {sorted_missing[:5]}"
                    f"{'...' if len(units_missing_fe) > 5 else ''})"
                )
            if periods_missing_fe:
                sorted_missing = sorted(periods_missing_fe)
                parts.append(
                    f"{len(periods_missing_fe)} post-treatment period(s) have no "
                    f"untreated units (periods: {sorted_missing[:5]}"
                    f"{'...' if len(periods_missing_fe) > 5 else ''})"
                )
            msg = (
                "Rank condition violated: "
                + "; ".join(parts)
                + ". Affected treatment effects will be NaN."
            )
            if self.rank_deficient_action == "error":
                raise ValueError(msg)
            elif self.rank_deficient_action == "warn":
                warnings.warn(msg, UserWarning, stacklevel=2)

        # ---- Residualize ALL observations ----
        y_tilde = self._residualize(
            df, outcome, unit, time, covariates, unit_fe, time_fe, grand_mean, delta_hat
        )
        df["_y_tilde"] = y_tilde

        # ---- Stage 2: OLS of y_tilde on treatment indicators ----
        # Build design matrices and compute effects + GMM variance
        ref_period = -1 - self.anticipation

        # Survey degrees of freedom for t-distribution inference
        _survey_df = resolved_survey.df_survey if resolved_survey is not None else None
        # Replicate df: rank-deficient → NaN inference
        if _uses_replicate_ts and _survey_df is None:
            _survey_df = 0

        # Kit df-provenance SEED (M-022): the exact value the stage-2
        # aggregators below receive, captured BEFORE the replicate override
        # can rebind _survey_df — post-fit recompute must re-seed from it.
        _survey_df_stage2 = _survey_df

        # Wave E.3 parity (PR #482 SpilloverDiD precedent): under the survey
        # path, `score_pad_mask_arg` is the FULL-DOMAIN keep_mask identifying
        # rows present in the stage-1 / stage-2 OLS sample after the
        # always-treated drop. `cluster_ids_full` is the FULL-DOMAIN
        # post-injection PSU labels. Both are passed through to
        # `_compute_gmm_variance`, which zero-pads per-cluster scores onto
        # the full-domain PSU list before stratified-meat dispatch so
        # `n_psu` / `n_strata` / `df_survey` reflect the full design.
        # Replicate-variance fits do NOT pad — Replicate refits per-replicate
        # already handle the resampling at the survey-design level.
        # We gate the padding kwargs on `n_always_treated > 0` so the
        # zero-pad branch in `_compute_gmm_variance` (np.unique +
        # np.searchsorted + full-size c_by_cluster / s2_by_cluster copies)
        # only fires when always-treated rows were actually dropped —
        # baseline survey fits with no drop pass `None / None` and take
        # the bit-identical pre-PR path.
        _wave_e3_pad_active = (
            resolved_survey is not None and not _uses_replicate_ts and n_always_treated > 0
        )
        score_pad_mask_arg = keep_mask.values if _wave_e3_pad_active else None
        cluster_ids_full_arg = cluster_ids_full if _wave_e3_pad_active else None

        # Always compute overall ATT (static specification)
        overall_att, overall_se = self._stage2_static(
            df=df,
            unit=unit,
            time=time,
            first_treat=first_treat,
            covariates=covariates,
            omega_0_mask=omega_0_mask,
            omega_1_mask=omega_1_mask,
            unit_fe=unit_fe,
            time_fe=time_fe,
            grand_mean=grand_mean,
            delta_hat=delta_hat,
            cluster_var=cluster_var,
            kept_cov_mask=kept_cov_mask,
            survey_weights=survey_weights,
            survey_weight_type=survey_weight_type,
            resolved_survey=(resolved_survey if not _uses_replicate_ts else None),
            score_pad_mask=score_pad_mask_arg,
            cluster_ids_full=cluster_ids_full_arg,
        )

        # Compute overall ATT inference (may be overridden by replicate below)
        overall_t, overall_p, overall_ci = safe_inference(
            overall_att, overall_se, alpha=self.alpha, df=_survey_df
        )

        # Event study and group aggregation (full-sample, for point estimates)
        event_study_effects = None
        group_effects = None
        _es_vcov_ts: Optional[np.ndarray] = None
        _es_vcov_index_ts: Optional[List[int]] = None

        if aggregate in ("event_study", "all"):
            event_study_effects, _es_vcov_ts, _es_vcov_index_ts = self._stage2_event_study(
                df=df,
                unit=unit,
                time=time,
                first_treat=first_treat,
                covariates=covariates,
                omega_0_mask=omega_0_mask,
                omega_1_mask=omega_1_mask,
                unit_fe=unit_fe,
                time_fe=time_fe,
                grand_mean=grand_mean,
                delta_hat=delta_hat,
                cluster_var=cluster_var,
                treatment_groups=treatment_groups,
                ref_period=ref_period,
                balance_e=balance_e,
                kept_cov_mask=kept_cov_mask,
                survey_weights=survey_weights,
                survey_weight_type=survey_weight_type,
                survey_df=_survey_df,
                resolved_survey=(resolved_survey if not _uses_replicate_ts else None),
                score_pad_mask=score_pad_mask_arg,
                cluster_ids_full=cluster_ids_full_arg,
            )

        if aggregate in ("group", "all"):
            group_effects = self._stage2_group(
                df=df,
                unit=unit,
                time=time,
                first_treat=first_treat,
                covariates=covariates,
                omega_0_mask=omega_0_mask,
                omega_1_mask=omega_1_mask,
                unit_fe=unit_fe,
                time_fe=time_fe,
                grand_mean=grand_mean,
                delta_hat=delta_hat,
                cluster_var=cluster_var,
                treatment_groups=treatment_groups,
                kept_cov_mask=kept_cov_mask,
                survey_weights=survey_weights,
                survey_weight_type=survey_weight_type,
                survey_df=_survey_df,
                resolved_survey=(resolved_survey if not _uses_replicate_ts else None),
                score_pad_mask=score_pad_mask_arg,
                cluster_ids_full=cluster_ids_full_arg,
            )

        # Replicate variance override: derive keys from actual outputs, then refit
        _n_valid_rep_ts = None
        _vcov_rep_ts = None
        if _uses_replicate_ts:
            (
                _vcov_rep_ts,
                _n_valid_rep_ts,
                _survey_df,
            ) = self._replay_replicate_inference(
                df=df,
                outcome=outcome,
                unit=unit,
                time=time,
                first_treat=first_treat,
                covariates=covariates,
                omega_0_mask=omega_0_mask,
                omega_1_mask=omega_1_mask,
                cluster_var=cluster_var,
                treatment_groups=treatment_groups,
                ref_period=ref_period,
                balance_e=balance_e,
                keep_mask=keep_mask,
                resolved_survey=resolved_survey,
                overall_att=overall_att,
                event_study_effects=event_study_effects,
                group_effects=group_effects,
                survey_df_seed=_survey_df,
            )
            overall_se = float(np.sqrt(max(_vcov_rep_ts[0, 0], 0.0)))
            if survey_metadata is not None:
                survey_metadata.df_survey = _survey_df if _survey_df and _survey_df > 0 else None
            # Recompute overall inference with replicate SE/df
            overall_t, overall_p, overall_ci = safe_inference(
                overall_att, overall_se, alpha=self.alpha, df=_survey_df
            )

        # Build treatment effects DataFrame
        treated_df = df.loc[omega_1_mask, [unit, time, "_y_tilde", "_rel_time"]].copy()
        treated_df = treated_df.rename(columns={"_y_tilde": "tau_hat", "_rel_time": "rel_time"})
        tau_finite = treated_df["tau_hat"].notna() & np.isfinite(treated_df["tau_hat"].values)
        n_valid_te = int(tau_finite.sum())
        if n_valid_te > 0:
            if survey_weights is not None:
                treated_sw = survey_weights[omega_1_mask.values]
                sw_finite = np.where(tau_finite, treated_sw, 0.0)
                sw_sum = sw_finite.sum()
                treated_df["weight"] = sw_finite / sw_sum if sw_sum > 0 else 0.0
            else:
                treated_df["weight"] = np.where(tau_finite, 1.0 / n_valid_te, 0.0)
        else:
            treated_df["weight"] = 0.0

        # ---- Bootstrap ----
        bootstrap_results = None
        if self.n_bootstrap > 0:
            try:
                bootstrap_results = self._run_bootstrap(
                    df=df,
                    unit=unit,
                    time=time,
                    first_treat=first_treat,
                    covariates=covariates,
                    omega_0_mask=omega_0_mask,
                    omega_1_mask=omega_1_mask,
                    unit_fe=unit_fe,
                    time_fe=time_fe,
                    grand_mean=grand_mean,
                    delta_hat=delta_hat,
                    cluster_var=cluster_var,
                    kept_cov_mask=kept_cov_mask,
                    treatment_groups=treatment_groups,
                    ref_period=ref_period,
                    balance_e=balance_e,
                    original_att=overall_att,
                    original_event_study=event_study_effects,
                    original_group=group_effects,
                    aggregate=aggregate,
                    resolved_survey=resolved_survey,
                )
            except NotImplementedError:
                raise  # Don't swallow explicit rejections (e.g. lonely_psu="adjust")
            except Exception as e:
                warnings.warn(
                    f"Bootstrap failed: {e}. Skipping bootstrap inference.",
                    UserWarning,
                    stacklevel=2,
                )

            if bootstrap_results is not None:
                # Update inference with bootstrap results
                overall_se = bootstrap_results.overall_att_se
                overall_t = (
                    overall_att / overall_se
                    if np.isfinite(overall_se) and overall_se > 0
                    else np.nan
                )
                overall_p = bootstrap_results.overall_att_p_value
                overall_ci = bootstrap_results.overall_att_ci

                # Update event study
                if event_study_effects and bootstrap_results.event_study_ses:
                    for h in event_study_effects:
                        if (
                            h in bootstrap_results.event_study_ses
                            and event_study_effects[h].get("n_obs", 1) > 0
                        ):
                            event_study_effects[h]["se"] = bootstrap_results.event_study_ses[h]
                            assert bootstrap_results.event_study_cis is not None
                            event_study_effects[h]["conf_int"] = bootstrap_results.event_study_cis[
                                h
                            ]
                            assert bootstrap_results.event_study_p_values is not None
                            event_study_effects[h]["p_value"] = (
                                bootstrap_results.event_study_p_values[h]
                            )
                            eff_val = event_study_effects[h]["effect"]
                            se_val = event_study_effects[h]["se"]
                            event_study_effects[h]["t_stat"] = safe_inference(
                                eff_val, se_val, alpha=self.alpha
                            )[0]

                # Update group effects
                if group_effects and bootstrap_results.group_ses:
                    for g in group_effects:
                        if g in bootstrap_results.group_ses:
                            group_effects[g]["se"] = bootstrap_results.group_ses[g]
                            assert bootstrap_results.group_cis is not None
                            group_effects[g]["conf_int"] = bootstrap_results.group_cis[g]
                            assert bootstrap_results.group_p_values is not None
                            group_effects[g]["p_value"] = bootstrap_results.group_p_values[g]
                            eff_val = group_effects[g]["effect"]
                            se_val = group_effects[g]["se"]
                            group_effects[g]["t_stat"] = safe_inference(
                                eff_val, se_val, alpha=self.alpha
                            )[0]

        # Resolve cluster_name / n_clusters for Results metadata. Suppress under
        # ANY survey design (the summary survey block already reports the
        # design's PSU/strata/replicate metadata; replicate-weight variance
        # ignores cluster entirely). Otherwise count clusters EXACTLY the way the
        # GMM sandwich does — `np.unique(df[cluster_var].values)` — so the
        # reported G can never disagree with the SE:
        #   - `df` (not the full input `data`) excludes always-treated units
        #     dropped above at `df = df[~df[unit].isin(always_treated_units)]`,
        #     matching the post-drop `cluster_ids=df[cluster_var].values` fed to
        #     `_compute_gmm_variance`;
        #   - `np.unique` (not `Series.nunique()`, which drops NaN) keeps the
        #     same single NaN group the variance forms, so missing cluster IDs
        #     cannot make the metadata undercount.
        # `cluster_var` is `self.cluster`, or the `unit` column when
        # `cluster=None` (the Gardner sandwich always clusters at unit by
        # default), so the summary renders the unit-cluster CR1 label, not HC1.
        if resolved_survey is not None:
            _cluster_name_for_results: Optional[str] = None
            _n_clusters_for_results: Optional[int] = None
        else:
            _cluster_name_for_results = self.cluster if self.cluster is not None else unit
            _n_clusters_for_results = int(np.unique(df[cluster_var].values).size)

        # Event-study VCV / df persistence, gated by inference mode (spec
        # section 5, row M-092). Persist the analytical GMM matrix only when
        # the stored ES SEs are actually its diagonal:
        # - replicate-weight survey: reported SEs come from _vcov_rep_ts's
        #   [overall, ES, groups] layout (filtered/Prop-5 horizons absent) -
        #   ship vcov=None (CS precedent) but thread the final _survey_df
        #   that every recomputed ES row's safe_inference used;
        # - bootstrap: percentile p/CIs replaced the analytical inference and
        #   no bootstrap covariance exists - clear both vcov and df;
        # - otherwise: persist V + est_horizons; the scalar _survey_df (None
        #   on non-survey fits -> normal theory -> no df recorded) is the df
        #   every estimated row's safe_inference received.
        _es_df_scalar: Optional[float] = (
            float(_survey_df)
            if _survey_df is not None and np.isfinite(_survey_df) and _survey_df > 0
            else None
        )
        if _uses_replicate_ts:
            _es_vcov_final: Optional[np.ndarray] = None
            _es_vcov_index_final: Optional[List[int]] = None
            _es_df_final = _es_df_scalar
        elif bootstrap_results is not None:
            _es_vcov_final = None
            _es_vcov_index_final = None
            _es_df_final = None
        else:
            _es_vcov_final = _es_vcov_ts
            _es_vcov_index_final = _es_vcov_index_ts
            _es_df_final = _es_df_scalar
        if event_study_effects is None:
            _es_vcov_final = None
            _es_vcov_index_final = None
            _es_df_final = None

        # Construct results
        self.results_ = TwoStageDiDResults(
            treatment_effects=treated_df,
            overall_att=overall_att,
            overall_se=overall_se,
            overall_t_stat=overall_t,
            overall_p_value=overall_p,
            overall_conf_int=overall_ci,
            event_study_effects=event_study_effects,
            group_effects=group_effects,
            groups=treatment_groups,
            time_periods=time_periods,
            n_obs=len(df),
            n_treated_obs=n_omega_1,
            n_untreated_obs=n_omega_0,
            n_treated_units=n_treated_units,
            n_control_units=n_control_units,
            alpha=self.alpha,
            anticipation=self.anticipation,
            bootstrap_results=bootstrap_results,
            survey_metadata=survey_metadata,
            vcov_type=self.vcov_type,
            cluster_name=_cluster_name_for_results,
            n_clusters=_n_clusters_for_results,
            event_study_vcov=_es_vcov_final,
            event_study_vcov_index=_es_vcov_index_final,
            event_study_df=_es_df_final,
        )

        # Attach the post-fit aggregation kit (M-022/M-119). Unconditional —
        # including bootstrap fits, whose gate lives in _aggregate_compute
        # (a FAILED bootstrap leaves bootstrap_results=None and the fit
        # aggregates normally).
        self.results_._aggregation_kit = _build_twostage_aggregation_kit(
            df=df,
            outcome=outcome,
            unit=unit,
            time=time,
            first_treat=first_treat,
            covariates=covariates,
            omega_0_mask=omega_0_mask,
            omega_1_mask=omega_1_mask,
            unit_fe=unit_fe,
            time_fe=time_fe,
            grand_mean=grand_mean,
            delta_hat=delta_hat,
            kept_cov_mask=kept_cov_mask,
            cluster_var=cluster_var,
            treatment_groups=treatment_groups,
            ref_period=ref_period,
            survey_weights=survey_weights,
            survey_weight_type=survey_weight_type,
            resolved_survey=resolved_survey,
            uses_replicate=_uses_replicate_ts,
            keep_mask=keep_mask,
            score_pad_mask=score_pad_mask_arg,
            cluster_ids_full=cluster_ids_full_arg,
            overall_att=overall_att,
            n_treated_obs=n_omega_1,
            survey_df_stage2=_survey_df_stage2,
            survey_df_final=_survey_df,
            survey_metadata=survey_metadata,
            pretrends=self.pretrends,
            horizon_max=self.horizon_max,
            rank_deficient_action=self.rank_deficient_action,
            alpha=self.alpha,
            anticipation=self.anticipation,
        )

        self.is_fitted_ = True
        return self.results_

    # =========================================================================
    # Stage 1: OLS on untreated observations
    # =========================================================================

    # =========================================================================
    # Residualization
    # =========================================================================

    # =========================================================================
    # Stage 2 specifications
    # =========================================================================

    # =========================================================================
    # GMM score computation
    # =========================================================================

    # =========================================================================
    # GMM Sandwich Variance (Butts & Gardner 2022)
    # =========================================================================

    # =========================================================================
    # sklearn-compatible interface
    # =========================================================================

    @staticmethod
    def _validate_vcov_type(vcov_type: str) -> None:
        """Validate ``vcov_type`` against TwoStageDiD's narrow GMM-sandwich
        variance contract.

        Called from ``__init__`` AND ``fit()`` so sklearn-style
        ``set_params(vcov_type=...)`` mutations are re-checked at use time rather
        than silently accepted by the setter (mirrors the ImputationDiD /
        TripleDifference / CallawaySantAnna pattern). TwoStageDiD's variance is
        the Gardner (2022) two-stage GMM cluster-sandwich
        (``V = bread @ (S' S) @ bread`` with the per-cluster GMM-corrected score
        ``S_g = gamma_hat' c_g - X_2g' eps_2g``); the contract is permanently
        narrow to ``{"hc1"}``.
        """
        _accepted_vcov = {"hc1"}
        _sandwich_incompatible = {"classical", "hc2", "hc2_bm"}
        _deferred_vcov = {"conley"}

        if vcov_type in _sandwich_incompatible:
            raise ValueError(
                f"TwoStageDiD(vcov_type={vcov_type!r}) is rejected: TwoStageDiD "
                "uses the Gardner (2022) two-stage GMM sandwich, whose meat is "
                "the per-cluster GMM-corrected score "
                "S_g = gamma_hat' c_g - X_2g' eps_2g, which folds first-stage FE "
                "estimation uncertainty into the score via the gamma_hat' c_g "
                "term. Hat-matrix leverage (hc2) and Bell-McCaffrey "
                "Satterthwaite DOF (hc2_bm) are defined for textbook "
                "single-equation OLS residuals; there is no single hat matrix "
                "spanning both stages, and the Gardner first-stage correction "
                "has not been derived for the leverage-corrected or "
                "homoskedastic (classical) meat structures (no reference "
                "implementation — clubSandwich covers single-equation WLS/OLS "
                "CR2, not two-stage GMM). Use vcov_type='hc1' (the default) with "
                "cluster=<col> for cluster-robust inference."
            )
        if vcov_type in _deferred_vcov:
            raise ValueError(
                f"TwoStageDiD(vcov_type={vcov_type!r}) is not yet supported: "
                "TwoStageDiD's GMM sandwich (_compute_gmm_variance) has no "
                "spatial-HAC path today (the Conley machinery lives in the "
                "separate SpilloverDiD helper). See DEFERRED.md for the deferred "
                "follow-up row. Use vcov_type='hc1' (the default) with "
                "cluster=<col> for cluster-robust inference."
            )
        if vcov_type not in _accepted_vcov:
            raise ValueError(
                f"TwoStageDiD(vcov_type={vcov_type!r}) is invalid. "
                f"Accepted: {sorted(_accepted_vcov)}."
            )

    # get_params/set_params come from BaseEstimator.

    def summary(self) -> str:
        """Get summary of estimation results."""
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before calling summary()")
        assert self.results_ is not None
        return self.results_.summary()

    def print_summary(self) -> None:
        """Print summary to stdout."""
        print(self.summary())


# =============================================================================
# Post-fit aggregation kit (M-022/M-119)
# =============================================================================


def _build_twostage_aggregation_kit(
    *,
    df: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    first_treat: str,
    covariates: Optional[List[str]],
    omega_0_mask: pd.Series,
    omega_1_mask: pd.Series,
    unit_fe: Dict[Any, float],
    time_fe: Dict[Any, float],
    grand_mean: float,
    delta_hat: Optional[np.ndarray],
    kept_cov_mask: Optional[np.ndarray],
    cluster_var: str,
    treatment_groups: List[Any],
    ref_period: int,
    survey_weights: Optional[np.ndarray],
    survey_weight_type: str,
    resolved_survey: Any,
    uses_replicate: bool,
    keep_mask: pd.Series,
    score_pad_mask: Optional[np.ndarray],
    cluster_ids_full: Optional[np.ndarray],
    overall_att: float,
    n_treated_obs: int,
    survey_df_stage2: Optional[int],
    survey_df_final: Optional[int],
    survey_metadata: Optional[Any],
    pretrends: bool,
    horizon_max: Optional[int],
    rank_deficient_action: str,
    alpha: float,
    anticipation: int,
) -> AggregationKit:
    """Build the PANEL-BACKED post-fit aggregation kit (rows M-022/M-119).

    TwoStageDiD's event-study/group aggregation is a fresh Stage-2 OLS +
    joint GMM sandwich per level — no compact influence payload can honor
    a different ``balance_e`` post-fit — so the kit retains a
    COLUMN-SUBSET COPY of the working frame (only the columns the moved
    methods read by name, deduplicated: ``cluster=`` may legally name the
    unit/time/first_treat column) plus the Stage-1 FE model and survey
    objects. This is the FIRST new panel retention on TwoStageDiD results
    (a documented memory-contract change, O(n_obs); replicate designs
    additionally retain the (n_obs x R) replicate matrix via
    ``resolved_survey`` — the identifier-minimization guarantee of the
    CS/EDiD kits deliberately does NOT hold here).

    ``score_pad_mask``/``cluster_ids_full`` are the Wave-E.3-GATED values
    fit actually passed (None unless the always-treated pad is active) —
    storing the ungated arrays would fire the zero-pad branch on baseline
    survey fits and break inertness.

    Value snapshots isolate recompute from public-field mutation
    (``treatment_groups`` copied; ``survey_metadata`` a
    ``dataclasses.replace`` copy the ES carrier builds from). df channels:
    ``survey_df_stage2`` (what the stage-2 aggregators received — the
    recompute seed), ``survey_df_final`` (what the stored overall
    inference received — the 'simple' relay df), and the metadata copy's
    own ``df_survey`` (the fit-final container channel).

    ``influence`` is EMPTY BY DESIGN: the recompute is panel-backed
    (scores re-derived from the retained frame), not IF-payload-backed.
    """
    subset_cols = list(
        dict.fromkeys(
            [unit, time, outcome, first_treat]
            + list(covariates or [])
            + [cluster_var]
            + (["_survey_cluster"] if "_survey_cluster" in df.columns else [])
            + ["_never_treated", "_rel_time", "_y_tilde"]
        )
    )
    bookkeeping: Dict[str, Any] = {
        # Panel-backed working objects (column-subset copy; index/row order
        # preserved so numerics are identical)
        "df": df[subset_cols].copy(),
        "outcome": outcome,
        "unit": unit,
        "time": time,
        "first_treat": first_treat,
        "covariates": list(covariates) if covariates else None,
        "omega_0_mask": omega_0_mask,
        "omega_1_mask": omega_1_mask,
        "unit_fe": unit_fe,
        "time_fe": time_fe,
        "grand_mean": grand_mean,
        "delta_hat": delta_hat,
        "kept_cov_mask": kept_cov_mask,
        "cluster_var": cluster_var,
        "treatment_groups": list(treatment_groups),
        "ref_period": ref_period,
        "survey_weights": survey_weights,
        "survey_weight_type": survey_weight_type,
        "resolved_survey": resolved_survey,
        "uses_replicate": bool(uses_replicate),
        "keep_mask": keep_mask,
        "score_pad_mask": score_pad_mask,
        "cluster_ids_full": cluster_ids_full,
        # Value snapshots (isolation from public-field / estimator mutation)
        "overall_att": float(overall_att),
        "n_treated_obs": int(n_treated_obs),
        "survey_df_stage2": survey_df_stage2,
        "survey_df_final": survey_df_final,
        "survey_metadata": (
            dataclasses.replace(survey_metadata) if survey_metadata is not None else None
        ),
        "pretrends": pretrends,
        "horizon_max": horizon_max,
        "rank_deficient_action": rank_deficient_action,
    }
    return AggregationKit(
        bookkeeping=bookkeeping,
        influence={},
        alpha=alpha,
        anticipation=anticipation,
        cband=False,  # no simultaneous-band concept on this estimator
        bootstrap=None,  # replay not wired; recompute levels fail closed ('simple' relays, M-027)
    )


# =============================================================================
# Convenience function
# =============================================================================


def two_stage_did(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    first_treat: str,
    covariates: Optional[List[str]] = None,
    aggregate: Any = NOT_SUPPLIED,
    balance_e: Any = NOT_SUPPLIED,
    survey_design: Optional["SurveyDesign"] = None,
    vcov_type: str = "hc1",
    **kwargs,
) -> TwoStageDiDResults:
    """
    Convenience function for two-stage DiD estimation.

    .. deprecated:: 3.9
        ``two_stage_did()`` is deprecated and will be removed in 4.0
        (row M-071). Construct the estimator instead:
        ``TwoStageDiD(...).fit(data, ...)``.

    This is a shortcut for creating a TwoStageDiD estimator and calling fit().

    Parameters
    ----------
    data : pd.DataFrame
        Panel data.
    outcome : str
        Outcome variable column name.
    unit : str
        Unit identifier column name.
    time : str
        Time period column name.
    first_treat : str
        Column indicating first treatment period (0 for never-treated).
    covariates : list of str, optional
        Covariate column names.
    aggregate : str, optional
        DEPRECATED (3.9, removed in 4.0; row M-022): forwarded to ``fit()``,
        which warns — aggregate post-fit via
        ``results.aggregate('event_study')`` instead. A plain wrapper call
        (kwarg not supplied) never fires the aggregate warning; since 3.9
        every wrapper call fires the M-071 wrapper-deprecation warning.
    balance_e : int, optional
        DEPRECATED (3.9, removed in 4.0; row M-119): forwarded to ``fit()``,
        which warns — moves onto ``results.aggregate('event_study',
        balance_e=...)``.
    survey_design : SurveyDesign, optional
        Survey design specification for design-based inference. Supports
        pweight only (aweight/fweight raise ValueError). Supports strata,
        PSU, and FPC for design-based GMM sandwich variance. Strata enters
        survey df for t-distribution inference.
        Both analytical (n_bootstrap=0) and bootstrap inference are supported.
    vcov_type : str, default="hc1"
        Variance estimator family; permanently narrow to ``{"hc1"}`` (the Gardner
        2022 two-stage GMM cluster-sandwich). Analytical-sandwich families
        ``{"classical", "hc2", "hc2_bm"}`` and ``"conley"`` are rejected. See
        :class:`TwoStageDiD`.
    **kwargs
        Additional keyword arguments passed to TwoStageDiD constructor.

    Returns
    -------
    TwoStageDiDResults
        Estimation results.

    Examples
    --------
    >>> from diff_diff import two_stage_did, generate_staggered_data
    >>> data = generate_staggered_data(seed=42)
    >>> results = two_stage_did(data, 'outcome', 'unit', 'period',
    ...                         'first_treat')
    >>> results.print_summary()
    >>> results.aggregate('event_study').summary()  # post-fit aggregation
    """
    warnings.warn(
        "two_stage_did() is deprecated and will be removed in 4.0; "
        "construct the estimator instead: TwoStageDiD(...).fit(data, ...).",
        FutureWarning,
        stacklevel=2,
    )
    est = TwoStageDiD(vcov_type=vcov_type, **kwargs)
    return est.fit(
        data,
        outcome=outcome,
        unit=unit,
        time=time,
        first_treat=first_treat,
        covariates=covariates,
        aggregate=aggregate,
        balance_e=balance_e,
        survey_design=survey_design,
    )
