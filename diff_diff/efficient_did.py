"""
Efficient Difference-in-Differences estimator.

Implements the ATT estimator from Chen, Sant'Anna & Xie (2025).
Without covariates, achieves the semiparametric efficiency bound via
closed-form within-group covariances.  With covariates, uses a doubly
robust path with sieve outcome regressions, sieve propensity ratios, and
kernel-smoothed conditional Omega*(X) (see class docstring for details).

Under PT-All the model is overidentified and EDiD exploits this for
tighter inference; under PT-Post it reduces to the standard
single-baseline estimator (Callaway-Sant'Anna).

The variance machinery is purely influence-function-based: per-unit EIF
values aggregate via ``sqrt(mean(EIF**2)/n)`` (unclustered, HC1-style),
Liang-Zeger CR1 on cluster-aggregated EIF (under ``cluster=``), or
Taylor Series Linearization on the combined IF (under ``survey_design=``).
Because the per-unit EIF aggregation has no equivalent single design
matrix, analytical-sandwich families ``{classical, hc2, hc2_bm}`` cannot
be defined and the ``vcov_type`` input contract is permanently narrow to
``{"hc1"}`` — see ``docs/methodology/REGISTRY.md`` "IF-based variance
estimators vs analytical-sandwich estimators" for the structural rationale.
"""

import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from diff_diff._base import BaseEstimator
from diff_diff.efficient_did_bootstrap import (
    EDiDBootstrapResults,
    EfficientDiDBootstrapMixin,
)

if TYPE_CHECKING:
    from diff_diff.survey import ResolvedSurveyDesign
from diff_diff.efficient_did_covariates import (
    OMEGA_RIDGE_DEFAULT,
    _silverman_bandwidth,
    compute_conditional_cells_tiled,
    compute_eif_cov,
    compute_generated_outcomes_cov,
    compute_omega_star_conditional,
    compute_per_unit_weights,
    estimate_inverse_propensity_sieve,
    estimate_outcome_regression,
    estimate_propensity_ratio_sieve,
)
from diff_diff.efficient_did_results import EfficientDiDResults, HausmanPretestResult
from diff_diff.efficient_did_weights import (
    _omega_star_nocov_gram,
    compute_efficient_weights,
    compute_eif_nocov,
    compute_generated_outcomes_nocov,
    compute_omega_star_nocov,
    enumerate_valid_triples,
)
from diff_diff.utils import safe_inference

# Re-export for convenience
__all__ = ["EfficientDiD", "EfficientDiDResults", "EDiDBootstrapResults"]


def _validate_and_build_cluster_mapping(
    df: pd.DataFrame,
    unit: str,
    cluster: str,
    all_units: list,
) -> Tuple[np.ndarray, int]:
    """Validate cluster column and build unit-to-cluster-index mapping.

    Checks: column exists, no NaN, per-unit constancy, >= 2 clusters.
    Returns (cluster_indices, n_clusters).
    """
    if cluster not in df.columns:
        raise ValueError(f"Cluster column '{cluster}' not found in data.")
    if df[cluster].isna().any():
        raise ValueError(f"Cluster column '{cluster}' contains missing values.")
    cluster_by_unit = df.groupby(unit)[cluster]
    if (cluster_by_unit.nunique() > 1).any():
        raise ValueError(
            f"Cluster column '{cluster}' varies within unit. "
            "Cluster assignment must be constant per unit."
        )
    cluster_col = cluster_by_unit.first().reindex(all_units).values
    unique_clusters = np.unique(cluster_col)
    n_clusters = len(unique_clusters)
    if n_clusters < 2:
        raise ValueError(f"Need at least 2 clusters for cluster-robust SEs, got {n_clusters}.")
    cluster_to_idx = {c: i for i, c in enumerate(unique_clusters)}
    indices = np.array([cluster_to_idx[c] for c in cluster_col])
    return indices, n_clusters


def _cluster_aggregate(
    eif_mat: np.ndarray,
    cluster_indices: np.ndarray,
    n_clusters: int,
) -> np.ndarray:
    """Sum EIF values within clusters and center.

    Parameters
    ----------
    eif_mat : ndarray, shape (n_units,) or (n_units, k)
        EIF values — 1-D for a single estimand, 2-D for multiple.
    cluster_indices : ndarray, shape (n_units,)
        Integer cluster assignment per unit.
    n_clusters : int
        Number of unique clusters.

    Returns
    -------
    ndarray, shape (n_clusters,) or (n_clusters, k)
        Centered cluster-level sums.
    """
    if eif_mat.ndim == 1:
        sums = np.bincount(cluster_indices, weights=eif_mat, minlength=n_clusters).astype(float)
    else:
        sums = np.column_stack(
            [
                np.bincount(cluster_indices, weights=eif_mat[:, j], minlength=n_clusters)
                for j in range(eif_mat.shape[1])
            ]
        ).astype(float)
    return sums - sums.mean(axis=0)


def _compute_se_from_eif(
    eif: np.ndarray,
    n_units: int,
    cluster_indices: Optional[np.ndarray] = None,
    n_clusters: Optional[int] = None,
) -> float:
    """SE from EIF values, optionally with cluster-robust correction.

    Without clusters: ``sqrt(mean(EIF^2) / n)``.
    With clusters: Liang-Zeger sandwich — aggregate EIF within clusters,
    center, and apply G/(G-1) small-sample correction.
    """
    if cluster_indices is not None and n_clusters is not None:
        centered = _cluster_aggregate(eif, cluster_indices, n_clusters)
        correction = n_clusters / (n_clusters - 1) if n_clusters > 1 else 1.0
        var = correction * np.sum(centered**2) / (n_units**2)
        return float(np.sqrt(max(var, 0.0)))
    return float(np.sqrt(np.mean(eif**2) / n_units))


def _hausman_quadratic_form(
    delta: np.ndarray,
    cov_post: np.ndarray,
    cov_all: np.ndarray,
) -> Tuple[float, int, float, int, bool]:
    """Hausman statistic from the event-study delta and the two ES covariances.

    Implements the Theorem A.1 test statistic of Chen, Sant'Anna & Xie (2025,
    arXiv:2506.17729v1).  The variance-difference matrix is

        V = aCov(ES_post) - aCov(ES_all) = cov_post - cov_all

    (restricted minus efficient, PSD under H0 because the efficient estimator has
    the smaller variance), and the statistic is ``H = delta' V^+ delta`` with
    ``delta = ES_post - ES_all``.  ``V`` is inverted by Moore-Penrose pseudoinverse
    and the number of strictly positive eigenvalues is used as the chi-square
    degrees of freedom -- a finite-sample safeguard for a non-PSD ``V`` that equals
    ``|E|`` (the number of post-treatment horizons) when ``V`` is well-conditioned.

    Parameters
    ----------
    delta : ndarray, shape (|E|,)
        Event-study difference ``ES_post - ES_all`` (restricted minus efficient).
    cov_post, cov_all : ndarray, shape (|E|, |E|)
        Estimator-scale covariances of the restricted (PT-Post) and efficient
        (PT-All) event-study vectors.

    Returns
    -------
    H : float
        The Hausman statistic (``max(delta' V^+ delta, 0)``); NaN if ``V`` is
        non-finite or has no positive eigenvalues.
    effective_rank : int
        Number of positive eigenvalues of ``V`` (the chi-square degrees of freedom).
    p_value : float
        Upper-tail ``chi2(effective_rank)`` p-value; NaN when ``H`` is NaN.
    n_negative : int
        Number of substantially negative eigenvalues of ``V`` (efficiency-reversal
        diagnostic).
    finite_ok : bool
        False when ``V`` contains non-finite entries.
    """
    from scipy.stats import chi2

    V = cov_post - cov_all
    if not np.all(np.isfinite(V)):
        return np.nan, 0, np.nan, 0, False

    eigvals = np.linalg.eigvalsh(V)
    max_eigval = float(np.max(np.abs(eigvals))) if len(eigvals) > 0 else 0.0
    tol = max(1e-10 * max_eigval, 1e-15)

    n_negative = int(np.sum(eigvals < -tol))
    effective_rank = int(np.sum(eigvals > tol))
    if effective_rank == 0:
        return np.nan, 0, np.nan, n_negative, True

    V_pinv = np.linalg.pinv(V, rcond=tol / max_eigval if max_eigval > 0 else 1e-10)
    H = max(float(delta @ V_pinv @ delta), 0.0)
    p_value = float(chi2.sf(H, df=effective_rank))
    return H, effective_rank, p_value, n_negative, True


class EfficientDiD(EfficientDiDBootstrapMixin, BaseEstimator):
    """Efficient DiD estimator (Chen, Sant'Anna & Xie 2025).

    Without covariates, achieves the semiparametric efficiency bound for
    ATT(g,t) using a closed-form estimator based on within-group sample
    means and covariances.

    With covariates, uses a doubly robust path: sieve-based propensity
    score ratios (Eq 4.1-4.2), sieve outcome regressions (polynomial
    basis, AIC/BIC order selection), sieve-estimated inverse propensities
    (algorithm step 4), and kernel-smoothed conditional Omega*(X) with
    per-unit efficient weights (Eq 3.12).  The DR property ensures
    consistency if either the outcome regression or the sieve propensity
    ratio is correctly specified; because all nuisances are sieves /
    kernel smoothers (the paper's flexible-nuisance specification), the
    covariate path attains the semiparametric efficiency bound under the
    paper's regularity conditions (see REGISTRY.md).

    Parameters
    ----------
    pt_assumption : str, default ``"all"``
        Parallel trends variant: ``"all"`` (overidentified, uses all
        pre-treatment periods and comparison groups) or ``"post"``
        (just-identified, single baseline, equivalent to CS).
    alpha : float, default 0.05
        Significance level.
    cluster : str or None
        Column name for cluster-robust SEs.  When set, analytical SEs
        use the Liang-Zeger clustered sandwich estimator on EIF values.
        With ``n_bootstrap > 0``, bootstrap weights are generated at the
        cluster level (all units in a cluster share the same weight).
    vcov_type : str, default ``"hc1"``
        Variance-estimator family. Permanently narrow to ``{"hc1"}`` per
        the Chen-Sant'Anna-Xie (2025) IF-based variance — analytical-sandwich
        families ``{classical, hc2, hc2_bm}`` and ``conley`` are rejected
        at ``__init__`` / ``set_params``. See REGISTRY.md for the
        methodology rationale (no single design matrix on which hat-matrix
        leverage or Bell-McCaffrey Satterthwaite DOF can be defined).
        Use ``cluster=<col>`` for Liang-Zeger CR1 on cluster-aggregated EIF;
        use ``survey_design=`` for Taylor Series Linearization on the
        combined IF.
    control_group : str, default ``"never_treated"``
        Which units serve as the comparison group:
        ``"never_treated"`` requires a never-treated cohort (raises if
        none exist); ``"last_cohort"`` reclassifies the latest treatment
        cohort as pseudo-never-treated and drops periods at
        ``t >= last_g - anticipation`` so the pseudo-control's
        pre-treatment window excludes anticipation-contaminated periods.
        Distinct from CallawaySantAnna's ``"not_yet_treated"`` — see
        REGISTRY.md for details.
    n_bootstrap : int, default 0
        Number of multiplier bootstrap iterations (0 = analytical only).
    bootstrap_weights : str, default ``"rademacher"``
        Bootstrap weight distribution.
    seed : int or None
        Random seed for reproducibility.
    anticipation : int, default 0
        Number of anticipation periods (shifts the effective treatment
        boundary forward by this amount). When combined with
        ``control_group="last_cohort"``, also trims the pseudo-control
        period set at ``t >= last_g - anticipation`` (see REGISTRY.md).
    sieve_k_max : int or None
        Maximum polynomial degree for the covariate-path sieves — the
        propensity-ratio, inverse-propensity, AND outcome-regression fits all
        use it. None = auto (``floor(n_pos^{1/5})`` over each group's
        positive-weight support ``n_pos`` — the raw group size when unweighted —
        a growing sieve with no fixed ceiling, bounded by ``n_basis < n_pos``;
        zero-weight survey rows do not affect order selection). Only
        used with covariates. ``sieve_k_max=1`` forces every covariate-path
        sieve (outcome regression and both propensity sieves) to degree 1: it
        recovers the pre-sieve linear-OLS *outcome regression* but also
        degree-1-constrains the propensity sieves, so it does not reproduce the
        exact pre-sieve estimator.
    sieve_criterion : str, default ``"bic"``
        Information criterion (``"aic"`` or ``"bic"``) for the order selection
        of all covariate-path sieves (propensity ratio, inverse propensity, and
        outcome regression).
    ratio_clip : float, default 20.0
        Clip sieve propensity ratios to ``[1/ratio_clip, ratio_clip]``.
    kernel_bandwidth : float or None
        Bandwidth for Gaussian kernel in conditional Omega* estimation.
        None = Silverman's rule-of-thumb (automatic).
    omega_ridge : float, default ``OMEGA_RIDGE_DEFAULT`` (1e-6)
        Relative ridge for the Omega* inversion behind the efficient
        weights: solves ``(Omega* + omega_ridge * max(trace/H, 0) * I) x = 1``
        instead of inverting the numerically singular Omega* that PT-All's
        telescoping overidentified moments produce. Stabilizes per-cell
        ATT(g,t) against floating-point-level input/BLAS changes (1-ulp
        stability ~1e-9 vs ~1e-4 for the legacy pseudoinverse) without
        changing overall-ATT bias/RMSE/coverage (see REGISTRY.md).
        ``omega_ridge=0`` restores the exact legacy inv/pinv code path
        bit-for-bit - including its per-cell condition-number warnings and
        the slow O(n^2 H^2) conditional-Omega* loops, so expect the legacy
        runtime as well.

    Examples
    --------
    >>> from diff_diff import EfficientDiD
    >>> edid = EfficientDiD(pt_assumption="all")
    >>> results = edid.fit(data, outcome="y", unit="id", time="t",
    ...                    first_treat="first_treat", aggregate="all")
    >>> results.print_summary()
    """

    def __init__(
        self,
        pt_assumption: str = "all",
        alpha: float = 0.05,
        cluster: Optional[str] = None,
        vcov_type: str = "hc1",
        control_group: str = "never_treated",
        n_bootstrap: int = 0,
        bootstrap_weights: str = "rademacher",
        seed: Optional[int] = None,
        anticipation: int = 0,
        sieve_k_max: Optional[int] = None,
        sieve_criterion: str = "bic",
        ratio_clip: float = 20.0,
        kernel_bandwidth: Optional[float] = None,
        omega_ridge: float = OMEGA_RIDGE_DEFAULT,
    ):
        self.pt_assumption = pt_assumption
        self.alpha = alpha
        self.cluster = cluster
        self.vcov_type = vcov_type
        self.control_group = control_group
        self.n_bootstrap = n_bootstrap
        self.bootstrap_weights = bootstrap_weights
        self.seed = seed
        self.anticipation = anticipation
        self.sieve_k_max = sieve_k_max
        self.sieve_criterion = sieve_criterion
        self.ratio_clip = ratio_clip
        self.kernel_bandwidth = kernel_bandwidth
        self.omega_ridge = omega_ridge
        self.is_fitted_ = False
        self.results_: Optional[EfficientDiDResults] = None
        self._unit_resolved_survey: Optional["ResolvedSurveyDesign"] = None
        self._validate_params()

    def _validate_params(self) -> None:
        """Validate constrained parameters."""
        if self.pt_assumption not in ("all", "post"):
            raise ValueError(f"pt_assumption must be 'all' or 'post', got '{self.pt_assumption}'")
        if self.control_group not in ("never_treated", "last_cohort"):
            raise ValueError(
                f"control_group must be 'never_treated' or 'last_cohort', "
                f"got '{self.control_group}'"
            )
        valid_weights = ("rademacher", "mammen", "webb")
        if self.bootstrap_weights not in valid_weights:
            raise ValueError(
                f"bootstrap_weights must be one of {valid_weights}, "
                f"got '{self.bootstrap_weights}'"
            )
        if self.sieve_criterion not in ("aic", "bic"):
            raise ValueError(
                f"sieve_criterion must be 'aic' or 'bic', got '{self.sieve_criterion}'"
            )
        if not (np.isfinite(self.ratio_clip) and self.ratio_clip > 1.0):
            raise ValueError(f"ratio_clip must be finite and > 1.0, got {self.ratio_clip}")
        if self.kernel_bandwidth is not None:
            if not (np.isfinite(self.kernel_bandwidth) and self.kernel_bandwidth > 0):
                raise ValueError(
                    f"kernel_bandwidth must be finite and > 0 (or None for auto), "
                    f"got {self.kernel_bandwidth}"
                )
        if self.sieve_k_max is not None:
            if not (isinstance(self.sieve_k_max, (int, np.integer)) and self.sieve_k_max > 0):
                raise ValueError(
                    f"sieve_k_max must be a positive integer (or None for auto), "
                    f"got {self.sieve_k_max}"
                )
        if not (np.isfinite(self.omega_ridge) and self.omega_ridge >= 0):
            raise ValueError(
                f"omega_ridge must be finite and >= 0 (0 = legacy inv/pinv path), "
                f"got {self.omega_ridge}"
            )
        self._validate_vcov_type(self.vcov_type)

    @staticmethod
    def _validate_vcov_type(vcov_type: str) -> None:
        """Validate ``vcov_type`` against EfficientDiD's narrow IF-based contract.

        Permanently accepts ``{"hc1"}`` only — EfficientDiD uses
        influence-function-based variance per Chen-Sant'Anna-Xie (2025) achieving
        the semiparametric efficiency bound. The per-unit EIF aggregation has no
        equivalent single design matrix, so analytical-sandwich families
        (``classical``, ``hc2``, ``hc2_bm``) cannot be defined; ``conley`` is
        deferred (see DEFERRED.md). Mirrors the narrow-contract pattern in
        :class:`ImputationDiD`, :class:`CallawaySantAnna`, and
        :class:`TripleDifference`.
        """
        _accepted_vcov = {"hc1"}
        _if_incompatible_vcov = {"classical", "hc2", "hc2_bm"}
        _deferred_vcov = {"conley"}

        if vcov_type in _if_incompatible_vcov:
            raise ValueError(
                f"EfficientDiD(vcov_type={vcov_type!r}) is rejected: "
                f"EfficientDiD uses influence-function-based variance per Chen, "
                f"Sant'Anna, and Xie (2025) achieving the semiparametric efficiency "
                f"bound for ATT(g,t). The per-unit EIF aggregation has no equivalent "
                f"single design matrix on which hat matrix leverage or Bell-McCaffrey "
                f"Satterthwaite DOF can be defined, so analytical-sandwich families "
                f"{{classical, hc2, hc2_bm}} are not paper-prescribed. Use "
                f"vcov_type='hc1' (the default) with cluster=<col> for the "
                f"Liang-Zeger clustered EIF sandwich estimator."
            )
        if vcov_type in _deferred_vcov:
            raise ValueError(
                f"EfficientDiD(vcov_type={vcov_type!r}) is not yet supported: "
                f"spatial-HAC composition with EIF aggregation has no reference "
                f"implementation today. See DEFERRED.md for the deferred follow-up row. "
                f"Use vcov_type='hc1' (the default) with cluster=<col> for "
                f"cluster-robust inference."
            )
        if vcov_type not in _accepted_vcov:
            raise ValueError(
                f"EfficientDiD(vcov_type={vcov_type!r}) is invalid. "
                f"Accepted: {sorted(_accepted_vcov)}."
            )

    # -- sklearn compatibility ------------------------------------------------

    # get_params/set_params come from BaseEstimator.

    # -- Main estimation ------------------------------------------------------

    def fit(
        self,
        data: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        first_treat: str,
        covariates: Optional[List[str]] = None,
        aggregate: Optional[str] = None,
        balance_e: Optional[int] = None,
        survey_design: Optional[Any] = None,
        store_eif: bool = False,
    ) -> EfficientDiDResults:
        """Fit the Efficient DiD estimator.

        Parameters
        ----------
        data : DataFrame
            Balanced panel data.
        outcome : str
            Outcome variable column name.
        unit : str
            Unit identifier column name.
        time : str
            Time period column name.
        first_treat : str
            Column indicating first treatment period.
            Use 0 or ``np.inf`` for never-treated units.
        covariates : list of str, optional
            Column names for time-invariant unit-level covariates.
            When provided, uses the doubly robust path (outcome regression
            + propensity score ratios).
        aggregate : str, optional
            ``None``, ``"simple"``, ``"event_study"``, ``"group"``, or
            ``"all"``.
        balance_e : int, optional
            Balance event study at this relative period.
        survey_design : SurveyDesign, optional
            Survey design specification for design-based inference.
            Applies survey weights to all means, covariances, and cohort
            fractions, and uses Taylor Series Linearization for SE
            estimation.  Cannot be combined with ``cluster``.
        store_eif : bool, default False
            Store per-(g,t) EIF vectors in the results object.  Used
            internally by :meth:`hausman_pretest`; not needed for
            normal usage.

        Returns
        -------
        EfficientDiDResults

        Raises
        ------
        ValueError
            Missing columns, unbalanced panel, non-absorbing treatment,
            or PT-Post without a never-treated group.
        """
        self._validate_params()

        if self.cluster is not None and survey_design is not None:
            raise NotImplementedError(
                "cluster and survey_design cannot both be set. "
                "Use survey_design with PSU/strata for cluster-robust inference."
            )

        # Resolve survey design if provided
        from diff_diff.survey import _resolve_survey_for_fit, build_unit_first_row_index

        resolved_survey, survey_weights, survey_weight_type, survey_metadata = (
            _resolve_survey_for_fit(survey_design, data, "analytical")
        )

        # Validate within-unit constancy for panel survey designs
        if resolved_survey is not None:
            from diff_diff.survey import _validate_unit_constant_survey

            _validate_unit_constant_survey(data, unit, survey_design)

        # Store survey df for safe_inference calls (t-distribution with survey df)
        self._survey_df = survey_metadata.df_survey if survey_metadata is not None else None
        # Guard: replicate design with undefined df → NaN inference
        if (
            self._survey_df is None
            and resolved_survey is not None
            and hasattr(resolved_survey, "uses_replicate_variance")
            and resolved_survey.uses_replicate_variance
        ):
            self._survey_df = 0

        # Bootstrap + survey supported via PSU-level multiplier bootstrap.

        # Normalize empty covariates list to None (use nocov path)
        if covariates is not None and len(covariates) == 0:
            covariates = None
        use_covariates = covariates is not None

        # ----- Validate inputs -----
        required_cols = [outcome, unit, time, first_treat]
        missing = [c for c in required_cols if c not in data.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        df = data.copy()
        df[time] = pd.to_numeric(df[time])
        df[first_treat] = pd.to_numeric(df[first_treat])

        # Normalize never-treated: inf -> 0 internally, keep track
        df["_never_treated"] = (df[first_treat] == 0) | (df[first_treat] == np.inf)
        df.loc[df[first_treat] == np.inf, first_treat] = 0

        time_periods = sorted(df[time].unique())
        treatment_groups = sorted([g for g in df[first_treat].unique() if g > 0])

        # Validate balanced panel
        unit_period_counts = df.groupby(unit)[time].nunique()
        n_periods = len(time_periods)
        if (unit_period_counts != n_periods).any():
            raise ValueError(
                "Unbalanced panel detected. EfficientDiD requires a balanced "
                "panel where every unit is observed in every time period."
            )

        # Reject non-finite outcomes (NaN/Inf corrupt Omega*/EIF calculations)
        non_finite_mask = ~np.isfinite(df[outcome])
        if non_finite_mask.any():
            n_bad = int(non_finite_mask.sum())
            raise ValueError(
                f"Found {n_bad} non-finite value(s) in outcome column '{outcome}'. "
                "EfficientDiD requires finite outcomes for all unit-period observations."
            )

        # Reject duplicate (unit, time) rows
        dup_mask = df.duplicated(subset=[unit, time], keep=False)
        if dup_mask.any():
            n_dups = int(dup_mask.sum())
            raise ValueError(
                f"Found {n_dups} duplicate ({unit}, {time}) rows. "
                "EfficientDiD requires exactly one observation per unit-period."
            )

        # Validate absorbing treatment (vectorized)
        ft_nunique = df.groupby(unit)[first_treat].nunique()
        bad_units = ft_nunique[ft_nunique > 1]
        if len(bad_units) > 0:
            uid = bad_units.index[0]
            raise ValueError(
                f"Non-absorbing treatment detected for unit {uid}: "
                "first_treat value changes over time."
            )

        # Unit info
        unit_info = (
            df.groupby(unit)
            .agg(
                {
                    first_treat: "first",
                    "_never_treated": "first",
                }
            )
            .reset_index()
        )
        n_treated_units = int((unit_info[first_treat] > 0).sum())
        n_control_units = int(unit_info["_never_treated"].sum())

        # Control group logic
        if self.control_group == "last_cohort":
            # Always reclassify last cohort as pseudo-control when requested
            if not treatment_groups:
                raise ValueError(
                    "No treated cohorts found. control_group='last_cohort' requires "
                    "at least 2 treatment cohorts."
                )
            last_g = max(treatment_groups)
            treatment_groups = [g for g in treatment_groups if g != last_g]
            if not treatment_groups:
                raise ValueError("Only one treatment cohort; cannot use last_cohort control.")
            effective_last = last_g - self.anticipation
            time_periods = [t for t in time_periods if t < effective_last]
            if len(time_periods) < 2:
                raise ValueError(
                    "Fewer than 2 time periods remain after trimming for last_cohort control."
                )
            unit_info.loc[unit_info[first_treat] == last_g, first_treat] = 0
            unit_info.loc[unit_info[first_treat] == 0, "_never_treated"] = True
            n_treated_units = int((unit_info[first_treat] > 0).sum())
            n_control_units = int(unit_info["_never_treated"].sum())
        elif n_control_units == 0:
            raise ValueError(
                "No never-treated units found. Use control_group='last_cohort' "
                "to use the last treatment cohort as a pseudo-control."
            )

        # ----- Prepare data -----
        all_units = sorted(df[unit].unique())
        n_units = len(all_units)

        # Build unit-to-first-panel-row index aligned to all_units (sorted)
        # order.  The previous approach (groupby cumcount == 0) yielded
        # first-appearance order which can differ from sorted order when the
        # input DataFrame is not pre-sorted by unit.
        self._unit_first_panel_row = build_unit_first_row_index(df[unit].values, all_units)

        # Build unit-level ResolvedSurveyDesign once (avoids repeated
        # construction in _compute_survey_eif_se and ensures consistent
        # unit-level df for safe_inference t-distribution).
        if resolved_survey is not None:
            self._unit_resolved_survey = resolved_survey.subset_to_units_by_row_idx(
                self._unit_first_panel_row
            )
            # Use unit-level df (not panel-level) for t-distribution
            self._survey_df = self._unit_resolved_survey.df_survey
            # Re-apply replicate guard: undefined df → NaN inference
            if self._survey_df is None and self._unit_resolved_survey.uses_replicate_variance:
                self._survey_df = 0
        else:
            self._unit_resolved_survey = None

        # Build cluster mapping if cluster-robust SEs requested
        if self.cluster is not None:
            unit_cluster_indices, n_clusters = _validate_and_build_cluster_mapping(
                df, unit, self.cluster, all_units
            )
            if n_clusters < 50:
                warnings.warn(
                    f"Only {n_clusters} clusters. Analytical clustered SEs may "
                    "be unreliable. Consider n_bootstrap > 0 for cluster "
                    "bootstrap inference.",
                    UserWarning,
                    stacklevel=2,
                )
        else:
            unit_cluster_indices = None
            n_clusters = None

        period_to_col = {p: i for i, p in enumerate(time_periods)}
        period_1 = time_periods[0]
        period_1_col = period_to_col[period_1]

        # Pivot outcome to wide matrix (n_units, n_periods)
        pivot = df.pivot(index=unit, columns=time, values=outcome)
        # Reindex to match all_units ordering and time_periods column order
        pivot = pivot.reindex(index=all_units, columns=time_periods)
        outcome_wide = pivot.values.astype(float)

        # Build cohort masks and fractions
        unit_info_indexed = unit_info.set_index(unit)
        unit_cohorts = unit_info_indexed.reindex(all_units)[first_treat].values.astype(
            float
        )  # 0 = never-treated

        cohort_masks: Dict[float, np.ndarray] = {}
        for g in treatment_groups:
            cohort_masks[g] = unit_cohorts == g
        never_treated_mask = unit_cohorts == 0
        cohort_masks[np.inf] = never_treated_mask  # also keyed by inf sentinel

        # ----- Unit-level survey weights -----
        # Survey weights in the panel are at obs level (unit x time).
        # EfficientDiD works at unit level.  Extract one weight per unit
        # by taking the first observation per unit (balanced panel, so
        # weights should be constant within unit).
        unit_level_weights: Optional[np.ndarray] = None
        if resolved_survey is not None:
            # Use the resolved survey's weights (already normalized per weight_type)
            # subset to unit level via _unit_first_panel_row (aligned to all_units)
            assert self._unit_resolved_survey is not None
            unit_level_weights = self._unit_resolved_survey.weights
        self._unit_level_weights = unit_level_weights

        cohort_fractions: Dict[float, float] = {}
        if unit_level_weights is not None:
            # Survey-weighted cohort fractions: sum(w_i for i in cohort) / sum(w_i)
            total_w = float(np.sum(unit_level_weights))
            for g in treatment_groups:
                cohort_fractions[g] = float(np.sum(unit_level_weights[cohort_masks[g]])) / total_w
            cohort_fractions[np.inf] = (
                float(np.sum(unit_level_weights[never_treated_mask])) / total_w
            )
        else:
            for g in treatment_groups:
                cohort_fractions[g] = float(np.sum(cohort_masks[g])) / n_units
            cohort_fractions[np.inf] = float(np.sum(never_treated_mask)) / n_units

        # ----- Small cohort warnings -----
        for g in treatment_groups:
            n_g = int(np.sum(cohort_masks[g]))
            frac_g = cohort_fractions[g]
            if n_g < 2:
                warnings.warn(
                    f"Cohort {g} has only {n_g} unit. Omega* inversion and "
                    "EIF computation may be numerically unstable.",
                    UserWarning,
                    stacklevel=2,
                )
            elif frac_g < 0.01:
                warnings.warn(
                    f"Cohort {g} represents {frac_g:.1%} of the sample (< 1%). "
                    "Efficient weights may be imprecise.",
                    UserWarning,
                    stacklevel=2,
                )

        # Guard: never-treated with zero survey weight → no valid comparisons
        # Applies to both covariates (DR nuisance) and nocov (weighted means) paths
        if cohort_fractions.get(np.inf, 0.0) <= 0 and unit_level_weights is not None:
            raise ValueError(
                "Never-treated group has zero survey weight. EfficientDiD "
                "requires a never-treated control group with positive "
                "survey weight for estimation."
            )

        # ----- Covariate preparation (if provided) -----
        covariate_matrix: Optional[np.ndarray] = None
        m_hat_cache: Dict[Tuple, np.ndarray] = {}
        r_hat_cache: Dict[Tuple[float, float], np.ndarray] = {}
        s_hat_cache: Dict[float, np.ndarray] = {}  # inverse propensities per group
        # Per-fit cache of the polynomial sieve basis, keyed (id(X), degree). The three
        # sieve nuisance helpers all build the basis from the same fit-level
        # `covariate_matrix`, so this shares each distinct degree's basis across them
        # instead of rebuilding it per helper. Lives only for this fit() call.
        sieve_basis_cache: Dict[Tuple[int, int], np.ndarray] = {}

        if use_covariates:
            assert covariates is not None  # for type narrowing

            # Validate covariate columns exist
            missing_cov = [c for c in covariates if c not in data.columns]
            if missing_cov:
                raise ValueError(f"Missing covariate columns: {missing_cov}")

            # Validate no NaN/Inf in covariates
            for col_name in covariates:
                non_finite_cov = ~np.isfinite(pd.to_numeric(df[col_name], errors="coerce"))
                if non_finite_cov.any():
                    n_bad = int(non_finite_cov.sum())
                    raise ValueError(
                        f"Found {n_bad} non-finite value(s) in covariate column "
                        f"'{col_name}'. Covariates must be finite."
                    )

            # Validate time-invariance: covariates must be constant within each unit
            for col_name in covariates:
                cov_nunique = df.groupby(unit)[col_name].nunique()
                varying = cov_nunique[cov_nunique > 1]
                if len(varying) > 0:
                    uid = varying.index[0]
                    raise ValueError(
                        f"Covariate '{col_name}' varies over time for unit {uid}. "
                        "EfficientDiD requires time-invariant covariates. "
                        "Extract base-period values before calling fit()."
                    )

            # Extract unit-level covariate matrix from period_1 observations
            base_df = df[df[time] == period_1].set_index(unit).reindex(all_units)
            covariate_matrix = base_df[list(covariates)].values.astype(float)

        # ----- Core estimation: ATT(g, t) for each target -----
        # Precompute per-group unit counts (avoid repeated np.sum in loop)
        n_treated_per_g = {g: int(np.sum(cohort_masks[g])) for g in treatment_groups}
        n_control_count = int(np.sum(never_treated_mask))

        group_time_effects: Dict[Tuple[Any, Any], Dict[str, Any]] = {}
        eif_by_gt: Dict[Tuple[Any, Any], np.ndarray] = {}
        stored_weights: Dict[Tuple[Any, Any], np.ndarray] = {}
        stored_cond: Dict[Tuple[Any, Any], float] = {}
        # Ridge path: ill-conditioned cells consolidate into ONE fit-level
        # warning (legacy omega_ridge=0 keeps the per-cell pinv warnings).
        _ill_conditioned_cells: List[Tuple[Any, Any, float]] = []
        # Ridge-path covariate cells are deferred to a fused unit-tiled pass 2
        # (compute_conditional_cells_tiled) after the sieve caches are fully
        # populated; each deferred cell holds an order-preserving placeholder
        # in group_time_effects until finalization.
        _cond_cell_specs: List[Dict[str, Any]] = []

        def _finalize_cell(g: Any, att_gt: float, eif_vals: np.ndarray) -> Dict[str, Any]:
            """Per-cell SE + inference record, shared by the inline and
            deferred (pass-2) paths.

            Analytical SE = sqrt(mean(EIF^2) / n)  [paper p.21]; with survey:
            TSL variance via compute_survey_vcov.
            """
            if self._unit_resolved_survey is not None:
                se_gt = self._compute_survey_eif_se(eif_vals)
            else:
                se_gt = _compute_se_from_eif(eif_vals, n_units, unit_cluster_indices, n_clusters)
            t_stat, p_val, ci = safe_inference(att_gt, se_gt, alpha=self.alpha, df=self._survey_df)
            return {
                "effect": att_gt,
                "se": se_gt,
                "t_stat": t_stat,
                "p_value": p_val,
                "conf_int": ci,
                "n_treated": n_treated_per_g[g],
                "n_control": n_control_count,
            }

        for g in treatment_groups:
            # Under PT-Post, use per-group baseline Y_{g-1-anticipation}
            # instead of the universal Y_1.  This implements the weaker
            # PT-Post assumption (parallel trends only from g-1 onward),
            # matching the Callaway-Sant'Anna estimator exactly.
            if self.pt_assumption == "post":
                effective_base = g - 1 - self.anticipation
                if effective_base not in period_to_col:
                    warnings.warn(
                        f"Cohort g={g} dropped: baseline period {effective_base} "
                        f"(g-1-anticipation) is not in the data.",
                        UserWarning,
                        stacklevel=2,
                    )
                    continue
                effective_p1_col = period_to_col[effective_base]
            else:
                effective_p1_col = period_1_col

            # Guard: skip cohorts with zero survey weight (all units zero-weighted)
            if cohort_fractions[g] <= 0:
                warnings.warn(
                    f"Cohort {g} has zero survey weight; skipping.",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            # Estimate all (g, t) cells including pre-treatment. Under PT-Post,
            # pre-treatment cells serve as placebo/pre-trend diagnostics, matching
            # the CallawaySantAnna implementation. Users filter to t >= g for
            # post-treatment effects; pre-treatment cells are clearly labeled by
            # their (g, t) coordinates in the results object.
            for t in time_periods:
                # Skip period_1 — it's the universal reference baseline,
                # not a target period
                if t == period_1:
                    continue

                # Enumerate valid comparison pairs
                pairs = enumerate_valid_triples(
                    target_g=g,
                    treatment_groups=treatment_groups,
                    time_periods=time_periods,
                    period_1=period_1,
                    pt_assumption=self.pt_assumption,
                    anticipation=self.anticipation,
                )

                # Filter out comparison pairs with zero survey weight
                if unit_level_weights is not None and pairs:
                    pairs = [
                        (gp, tpre)
                        for gp, tpre in pairs
                        if np.sum(
                            unit_level_weights[
                                never_treated_mask if np.isinf(gp) else cohort_masks[gp]
                            ]
                        )
                        > 0
                    ]

                # Ridge path: drop the degenerate same-cohort self-pair
                # (g' = g, t_pre = t), which arises only for PRE-treatment
                # cells (t < g). Its generated outcome telescopes to 0 = 0
                # identically (zero-information moment; the exact-null
                # direction of Omega*). The legacy pseudoinverse truncates
                # that direction, spreading weight over noisy moments; a
                # ridge would instead load ~all weight on the zero-variance
                # moment, collapsing pre-treatment placebos to a
                # deterministic 0 and silently disabling the pre-trend
                # diagnostic. Dropping the pair restores data-driven
                # placebos. NOT applied at omega_ridge=0 (bit-identical
                # legacy behavior).
                if self.omega_ridge > 0 and pairs:
                    pairs = [(gp, tpre) for gp, tpre in pairs if not (gp == g and tpre == t)]

                if not pairs:
                    warnings.warn(
                        f"No valid comparison pairs for (g={g}, t={t}). " "ATT will be NaN.",
                        UserWarning,
                        stacklevel=2,
                    )
                    t_stat, p_val, ci = np.nan, np.nan, (np.nan, np.nan)
                    group_time_effects[(g, t)] = {
                        "effect": np.nan,
                        "se": np.nan,
                        "t_stat": t_stat,
                        "p_value": p_val,
                        "conf_int": ci,
                        "n_treated": n_treated_per_g[g],
                        "n_control": n_control_count,
                    }
                    eif_by_gt[(g, t)] = np.zeros(n_units)
                    continue

                if use_covariates:
                    assert covariate_matrix is not None
                    t_col_val = period_to_col[t]

                    # Lazily populate nuisance caches for this (g, t)
                    for gp, tpre in pairs:
                        tpre_col_val = period_to_col[tpre]
                        # m_{inf, t, tpre}(X)
                        key_inf_t = (np.inf, t_col_val, tpre_col_val)
                        if key_inf_t not in m_hat_cache:
                            m_hat_cache[key_inf_t] = estimate_outcome_regression(
                                outcome_wide,
                                covariate_matrix,
                                never_treated_mask,
                                t_col_val,
                                tpre_col_val,
                                k_max=self.sieve_k_max,
                                criterion=self.sieve_criterion,
                                unit_weights=unit_level_weights,
                                basis_cache=sieve_basis_cache,
                            )
                        # m_{g', tpre, 1}(X)
                        key_gp_tpre = (gp, tpre_col_val, effective_p1_col)
                        if key_gp_tpre not in m_hat_cache:
                            gp_mask_for_reg = (
                                never_treated_mask if np.isinf(gp) else cohort_masks[gp]
                            )
                            m_hat_cache[key_gp_tpre] = estimate_outcome_regression(
                                outcome_wide,
                                covariate_matrix,
                                gp_mask_for_reg,
                                tpre_col_val,
                                effective_p1_col,
                                k_max=self.sieve_k_max,
                                criterion=self.sieve_criterion,
                                unit_weights=unit_level_weights,
                                basis_cache=sieve_basis_cache,
                            )
                        # r_{g, inf}(X) and r_{g, g'}(X) via sieve (Eq 4.1-4.2)
                        for comp in {np.inf, gp}:
                            rkey = (g, comp)
                            if rkey not in r_hat_cache:
                                comp_mask = (
                                    never_treated_mask if np.isinf(comp) else cohort_masks[comp]
                                )
                                r_hat_cache[rkey] = estimate_propensity_ratio_sieve(
                                    covariate_matrix,
                                    cohort_masks[g],
                                    comp_mask,
                                    k_max=self.sieve_k_max,
                                    criterion=self.sieve_criterion,
                                    ratio_clip=self.ratio_clip,
                                    unit_weights=unit_level_weights,
                                    basis_cache=sieve_basis_cache,
                                )

                    # Inverse propensity estimation (algorithm step 4)
                    # s_hat_{g'}(X) = 1/p_{g'}(X) for Eq 3.12 scaling.
                    # Populated BEFORE the ridge-path deferral so pass 2 sees
                    # complete caches (order swap vs the legacy sequence is
                    # inert: gen_out does not consume s_hat and vice versa).
                    for group_id in {g, np.inf} | {gp for gp, _ in pairs}:
                        if group_id not in s_hat_cache:
                            group_mask_s = (
                                never_treated_mask if np.isinf(group_id) else cohort_masks[group_id]
                            )
                            s_hat_cache[group_id] = estimate_inverse_propensity_sieve(
                                covariate_matrix,
                                group_mask_s,
                                k_max=self.sieve_k_max,
                                criterion=self.sieve_criterion,
                                unit_weights=unit_level_weights,
                                basis_cache=sieve_basis_cache,
                            )

                    if self.omega_ridge > 0:
                        # Fused tiled GEMM path: defer omega/weights/EIF to
                        # pass 2; placeholder preserves results ordering.
                        _cond_cell_specs.append(
                            {
                                "g": g,
                                "t": t,
                                "pairs": pairs,
                                "t_col": t_col_val,
                                "y1_col": effective_p1_col,
                            }
                        )
                        group_time_effects[(g, t)] = None
                        continue

                    # ----- Legacy (omega_ridge=0) covariate path -----
                    # Per-unit DR generated outcomes: shape (n_units, H)
                    gen_out = compute_generated_outcomes_cov(
                        target_g=g,
                        target_t=t,
                        valid_pairs=pairs,
                        outcome_wide=outcome_wide,
                        cohort_masks=cohort_masks,
                        never_treated_mask=never_treated_mask,
                        period_to_col=period_to_col,
                        period_1_col=effective_p1_col,
                        cohort_fractions=cohort_fractions,
                        m_hat_cache=m_hat_cache,
                        r_hat_cache=r_hat_cache,
                    )

                    y_hat = np.mean(gen_out, axis=0)  # shape (H,)

                    # Conditional Omega*(X) with per-unit propensities (Eq 3.12)
                    omega_cond = compute_omega_star_conditional(
                        target_g=g,
                        target_t=t,
                        valid_pairs=pairs,
                        outcome_wide=outcome_wide,
                        cohort_masks=cohort_masks,
                        never_treated_mask=never_treated_mask,
                        period_to_col=period_to_col,
                        period_1_col=effective_p1_col,
                        cohort_fractions=cohort_fractions,
                        covariate_matrix=covariate_matrix,
                        s_hat_cache=s_hat_cache,
                        bandwidth=self.kernel_bandwidth,
                        unit_weights=unit_level_weights,
                    )

                    # Per-unit weights: (n_units, H)
                    per_unit_w = compute_per_unit_weights(omega_cond, omega_ridge=self.omega_ridge)

                    # ATT = (survey-)weighted mean of per-unit DR scores
                    if per_unit_w.shape[1] > 0:
                        per_unit_scores = np.sum(per_unit_w * gen_out, axis=1)
                        if unit_level_weights is not None:
                            att_gt = float(np.average(per_unit_scores, weights=unit_level_weights))
                        else:
                            att_gt = float(np.mean(per_unit_scores))
                    else:
                        att_gt = np.nan

                    # EIF with per-unit weights (Remark 4.2: plug-in valid)
                    # Center on scalar ATT, not per-pair means (ensures mean(EIF) ≈ 0)
                    eif_vals = compute_eif_cov(per_unit_w, gen_out, att_gt, n_units)
                    eif_by_gt[(g, t)] = eif_vals
                else:
                    # No-covariates path (closed-form). Ridge path uses the
                    # Gram/GEMM twin of the O(H^2) _sample_cov loop
                    # (reassociation-level agreement); omega_ridge=0 keeps
                    # the legacy loop verbatim.
                    _omega_fn = (
                        _omega_star_nocov_gram if self.omega_ridge > 0 else compute_omega_star_nocov
                    )
                    omega = _omega_fn(
                        target_g=g,
                        target_t=t,
                        valid_pairs=pairs,
                        outcome_wide=outcome_wide,
                        cohort_masks=cohort_masks,
                        never_treated_mask=never_treated_mask,
                        period_to_col=period_to_col,
                        period_1_col=effective_p1_col,
                        cohort_fractions=cohort_fractions,
                        unit_weights=unit_level_weights,
                    )

                    weights, _, cond_num = compute_efficient_weights(
                        omega, omega_ridge=self.omega_ridge
                    )
                    stored_weights[(g, t)] = weights
                    if omega.size > 0:
                        stored_cond[(g, t)] = cond_num
                        if self.omega_ridge > 0 and cond_num > 1e12:
                            _ill_conditioned_cells.append((g, t, cond_num))

                    y_hat = compute_generated_outcomes_nocov(
                        target_g=g,
                        target_t=t,
                        valid_pairs=pairs,
                        outcome_wide=outcome_wide,
                        cohort_masks=cohort_masks,
                        never_treated_mask=never_treated_mask,
                        period_to_col=period_to_col,
                        period_1_col=effective_p1_col,
                        unit_weights=unit_level_weights,
                    )

                    att_gt = float(weights @ y_hat) if len(weights) > 0 else np.nan

                    eif_vals = compute_eif_nocov(
                        target_g=g,
                        target_t=t,
                        weights=weights,
                        valid_pairs=pairs,
                        outcome_wide=outcome_wide,
                        cohort_masks=cohort_masks,
                        never_treated_mask=never_treated_mask,
                        period_to_col=period_to_col,
                        period_1_col=effective_p1_col,
                        cohort_fractions=cohort_fractions,
                        n_units=n_units,
                        unit_weights=unit_level_weights,
                    )
                    eif_by_gt[(g, t)] = eif_vals

                group_time_effects[(g, t)] = _finalize_cell(g, att_gt, eif_vals)

        # ----- Pass 2: fused tiled conditional path (ridge, covariates) -----
        if _cond_cell_specs:
            assert covariate_matrix is not None
            bandwidth = (
                self.kernel_bandwidth
                if self.kernel_bandwidth is not None
                else _silverman_bandwidth(covariate_matrix, unit_level_weights)
            )
            cell_estimates = compute_conditional_cells_tiled(
                _cond_cell_specs,
                outcome_wide=outcome_wide,
                covariate_matrix=covariate_matrix,
                cohort_masks=cohort_masks,
                never_treated_mask=never_treated_mask,
                period_to_col=period_to_col,
                cohort_fractions=cohort_fractions,
                m_hat_cache=m_hat_cache,
                r_hat_cache=r_hat_cache,
                s_hat_cache=s_hat_cache,
                bandwidth=bandwidth,
                omega_ridge=self.omega_ridge,
                unit_weights=unit_level_weights,
            )
            for spec in _cond_cell_specs:
                g = spec["g"]
                t = spec["t"]
                att_gt, eif_vals = cell_estimates[(g, t)]
                eif_by_gt[(g, t)] = eif_vals
                group_time_effects[(g, t)] = _finalize_cell(g, att_gt, eif_vals)

        # Fit-level consolidation of ill-conditioned Omega* cells (ridge path).
        # Legacy (omega_ridge=0) warns per cell inside compute_efficient_weights.
        if _ill_conditioned_cells:
            _max_g, _max_t, _max_cond = max(_ill_conditioned_cells, key=lambda r: r[2])
            warnings.warn(
                f"Omega* was ill-conditioned (cond > 1e12) in "
                f"{len(_ill_conditioned_cells)} of {len(stored_cond)} (g, t) cells "
                f"(max cond {_max_cond:.2e} at (g={_max_g}, t={_max_t})); the "
                f"omega_ridge={self.omega_ridge:g} regularization handled these "
                f"cells (expected under PT-All, whose overidentified moment set "
                f"contains telescoping near-duplicate moments). Set omega_ridge=0 "
                f"to restore the legacy pseudoinverse path.",
                UserWarning,
                stacklevel=2,
            )

        if not group_time_effects:
            raise ValueError(
                "Could not estimate any group-time effects. "
                "Check data has sufficient observations."
            )

        # ----- Aggregation -----
        overall_att, overall_se = self._aggregate_overall(
            group_time_effects,
            eif_by_gt,
            n_units,
            cohort_fractions,
            unit_cohorts,
            cluster_indices=unit_cluster_indices,
            n_clusters=n_clusters,
        )
        overall_t, overall_p, overall_ci = safe_inference(
            overall_att, overall_se, alpha=self.alpha, df=self._survey_df
        )

        event_study_effects = None
        group_effects = None

        if aggregate in ("event_study", "all"):
            event_study_effects = self._aggregate_event_study(
                group_time_effects,
                eif_by_gt,
                n_units,
                cohort_fractions,
                treatment_groups,
                time_periods,
                balance_e,
                unit_cohorts=unit_cohorts,
                cluster_indices=unit_cluster_indices,
                n_clusters=n_clusters,
            )
        if aggregate in ("group", "all"):
            group_effects = self._aggregate_by_group(
                group_time_effects,
                eif_by_gt,
                n_units,
                cohort_fractions,
                treatment_groups,
                unit_cohorts=unit_cohorts,
                cluster_indices=unit_cluster_indices,
                n_clusters=n_clusters,
            )

        # ----- Bootstrap -----
        # Reject replicate-weight designs for bootstrap — replicate variance
        # is an analytical alternative, not compatible with bootstrap
        if (
            self.n_bootstrap > 0
            and self._unit_resolved_survey is not None
            and self._unit_resolved_survey.uses_replicate_variance
        ):
            raise NotImplementedError(
                "EfficientDiD bootstrap (n_bootstrap > 0) is not supported "
                "with replicate-weight survey designs. Replicate weights provide "
                "analytical variance; use n_bootstrap=0 instead."
            )
        bootstrap_results = None
        if self.n_bootstrap > 0 and eif_by_gt:
            bootstrap_results = self._run_multiplier_bootstrap(
                group_time_effects=group_time_effects,
                eif_by_gt=eif_by_gt,
                n_units=n_units,
                aggregate=aggregate,
                balance_e=balance_e,
                treatment_groups=treatment_groups,
                cohort_fractions=cohort_fractions,
                cluster_indices=unit_cluster_indices,
                n_clusters=n_clusters,
                resolved_survey=self._unit_resolved_survey,
                unit_level_weights=self._unit_level_weights,
            )
            # Update estimates with bootstrap inference
            overall_se = bootstrap_results.overall_att_se
            overall_t = safe_inference(overall_att, overall_se, alpha=self.alpha)[0]
            overall_p = bootstrap_results.overall_att_p_value
            overall_ci = bootstrap_results.overall_att_ci

            for gt in group_time_effects:
                if gt in bootstrap_results.group_time_ses:
                    group_time_effects[gt]["se"] = bootstrap_results.group_time_ses[gt]
                    group_time_effects[gt]["conf_int"] = bootstrap_results.group_time_cis[gt]
                    group_time_effects[gt]["p_value"] = bootstrap_results.group_time_p_values[gt]
                    eff = float(group_time_effects[gt]["effect"])
                    se = float(group_time_effects[gt]["se"])
                    group_time_effects[gt]["t_stat"] = safe_inference(eff, se, alpha=self.alpha)[0]

            es_cis = bootstrap_results.event_study_cis
            es_pvs = bootstrap_results.event_study_p_values
            if (
                event_study_effects is not None
                and bootstrap_results.event_study_ses is not None
                and es_cis is not None
                and es_pvs is not None
            ):
                for e in event_study_effects:
                    if e in bootstrap_results.event_study_ses:
                        event_study_effects[e]["se"] = bootstrap_results.event_study_ses[e]
                        event_study_effects[e]["conf_int"] = es_cis[e]
                        event_study_effects[e]["p_value"] = es_pvs[e]
                        eff = float(event_study_effects[e]["effect"])
                        se = float(event_study_effects[e]["se"])
                        event_study_effects[e]["t_stat"] = safe_inference(
                            eff, se, alpha=self.alpha
                        )[0]

            g_cis = bootstrap_results.group_effect_cis
            g_pvs = bootstrap_results.group_effect_p_values
            if (
                group_effects is not None
                and bootstrap_results.group_effect_ses is not None
                and g_cis is not None
                and g_pvs is not None
            ):
                for g in group_effects:
                    if g in bootstrap_results.group_effect_ses:
                        group_effects[g]["se"] = bootstrap_results.group_effect_ses[g]
                        group_effects[g]["conf_int"] = g_cis[g]
                        group_effects[g]["p_value"] = g_pvs[g]
                        eff = float(group_effects[g]["effect"])
                        se = float(group_effects[g]["se"])
                        group_effects[g]["t_stat"] = safe_inference(eff, se, alpha=self.alpha)[0]

        # ----- Build results -----
        self.results_ = EfficientDiDResults(
            group_time_effects=group_time_effects,
            overall_att=overall_att,
            overall_se=overall_se,
            overall_t_stat=overall_t,
            overall_p_value=overall_p,
            overall_conf_int=overall_ci,
            groups=treatment_groups,
            time_periods=time_periods,
            n_obs=n_units * len(time_periods),
            n_treated_units=n_treated_units,
            n_control_units=n_control_units,
            alpha=self.alpha,
            pt_assumption=self.pt_assumption,
            anticipation=self.anticipation,
            n_bootstrap=self.n_bootstrap,
            bootstrap_weights=self.bootstrap_weights,
            seed=self.seed,
            event_study_effects=event_study_effects,
            group_effects=group_effects,
            efficient_weights=stored_weights if stored_weights else None,
            omega_condition_numbers=stored_cond if stored_cond else None,
            control_group=self.control_group,
            # 2-branch cluster_name/n_clusters resolution: suppress under any
            # survey design (analytical TSL or replicate); populate under bare
            # ``cluster=``; default to None under unclustered, non-survey fits.
            # The default per-unit EIF SE ``sqrt(mean(EIF^2)/n)`` is HC1-style
            # (not auto-cluster-at-unit), so no third unit-default branch.
            cluster_name=(
                None
                if resolved_survey is not None
                else (self.cluster if self.cluster is not None else None)
            ),
            n_clusters=(
                None
                if resolved_survey is not None
                else (n_clusters if self.cluster is not None else None)
            ),
            vcov_type=self.vcov_type,
            influence_functions=eif_by_gt if store_eif else None,
            bootstrap_results=bootstrap_results,
            estimation_path="dr" if use_covariates else "nocov",
            sieve_k_max=self.sieve_k_max,
            sieve_criterion=self.sieve_criterion,
            ratio_clip=self.ratio_clip,
            kernel_bandwidth=self.kernel_bandwidth,
            omega_ridge=self.omega_ridge,
            survey_metadata=(
                self._recompute_unit_survey_metadata(survey_metadata)
                if survey_metadata is not None
                else None
            ),
        )
        self.is_fitted_ = True
        return self.results_

    def _recompute_unit_survey_metadata(self, panel_metadata):
        """Recompute survey metadata from unit-level design if available."""
        if self._unit_resolved_survey is not None:
            from diff_diff.survey import compute_survey_metadata

            meta = compute_survey_metadata(
                self._unit_resolved_survey,
                self._unit_resolved_survey.weights,
            )
            # Propagate effective replicate df if available
            # (but not the df=0 sentinel — keep metadata as None for undefined df)
            if (
                self._survey_df is not None
                and self._survey_df != 0
                and meta.df_survey != self._survey_df
            ):
                meta.df_survey = self._survey_df
            return meta
        return panel_metadata

    # -- Survey SE helpers ----------------------------------------------------

    def _compute_survey_eif_se(self, eif_vals: np.ndarray) -> float:
        """Compute SE from EIF scores using Taylor Series Linearization.

        Uses the pre-built unit-level ``_unit_resolved_survey`` constructed
        once in ``fit()``, ensuring consistent unit-level arrays and
        avoiding repeated subsetting of panel-level survey data.
        """
        # Built once in fit() before any call lands here (see docstring).
        assert self._unit_resolved_survey is not None
        if self._unit_resolved_survey.uses_replicate_variance:
            from diff_diff.survey import compute_replicate_if_variance

            # Score-scale IFs to match TSL bread: psi = w * eif / sum(w)
            w = self._unit_resolved_survey.weights
            psi_scaled = w * eif_vals / w.sum()
            variance, n_valid = compute_replicate_if_variance(
                psi_scaled, self._unit_resolved_survey
            )
            # Update survey df to reflect effective replicate count
            if n_valid < self._unit_resolved_survey.n_replicates:
                self._survey_df = n_valid - 1 if n_valid > 1 else None
            return float(np.sqrt(max(variance, 0.0))) if np.isfinite(variance) else np.nan

        from diff_diff.survey import compute_survey_vcov

        X_ones = np.ones((len(eif_vals), 1))
        vcov = compute_survey_vcov(X_ones, eif_vals, self._unit_resolved_survey)
        return float(np.sqrt(np.abs(vcov[0, 0])))

    def _eif_se(
        self,
        eif_vals: np.ndarray,
        n_units: int,
        cluster_indices: Optional[np.ndarray] = None,
        n_clusters: Optional[int] = None,
    ) -> float:
        """Compute SE from aggregated EIF scores.

        Dispatches to survey TSL when ``_unit_resolved_survey`` is set
        (during fit), otherwise uses cluster-robust or standard formula.
        """
        if self._unit_resolved_survey is not None:
            return self._compute_survey_eif_se(eif_vals)
        return _compute_se_from_eif(eif_vals, n_units, cluster_indices, n_clusters)

    # -- Aggregation helpers --------------------------------------------------

    def _compute_wif_contribution(
        self,
        keepers: List[Tuple],
        effects: np.ndarray,
        unit_cohorts: np.ndarray,
        cohort_fractions: Dict[float, float],
        n_units: int,
        unit_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute weight influence function correction (O(1) scale, matching EIF).

        This accounts for uncertainty in cohort-size aggregation weights.
        Matches R's ``did`` package WIF formula (staggered_aggregation.py:282-309),
        adapted to EDiD's EIF scale.

        Parameters
        ----------
        keepers : list of (g, t) tuples
            Post-treatment group-time pairs included in aggregation.
        effects : ndarray, shape (n_keepers,)
            ATT estimates for each keeper.
        unit_cohorts : ndarray, shape (n_units,)
            Cohort assignment for each unit (0 = never-treated).
        cohort_fractions : dict
            ``{cohort: n_cohort / n}`` for each cohort.
        n_units : int
            Total number of units.
        unit_weights : ndarray, shape (n_units,), optional
            Survey weights at the unit level.  When provided, uses the
            survey-weighted WIF formula: IF_i(p_g) = (w_i * 1{G_i=g} - pg_k).

        Returns
        -------
        ndarray, shape (n_units,)
            WIF contribution at O(1) scale, additive with ``agg_eif``.
        """
        groups_for_keepers = np.array([g for (g, t) in keepers])
        pg_keepers = np.array([cohort_fractions.get(g, 0.0) for g, t in keepers])
        sum_pg = pg_keepers.sum()
        if sum_pg == 0:
            return np.zeros(n_units)

        indicator = (unit_cohorts[:, None] == groups_for_keepers[None, :]).astype(float)

        if unit_weights is not None:
            # Survey-weighted WIF (matches staggered_aggregation.py:392-401):
            # IF_i(p_g) = (w_i * 1{G_i=g} - pg_k), NOT (1{G_i=g} - pg_k)
            weighted_indicator = indicator * unit_weights[:, None]
            indicator_diff = weighted_indicator - pg_keepers
            indicator_sum = np.sum(indicator_diff, axis=1)
        else:
            indicator_diff = indicator - pg_keepers
            indicator_sum = np.sum(indicator_diff, axis=1)

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            if1 = indicator_diff / sum_pg
            if2 = np.outer(indicator_sum, pg_keepers) / sum_pg**2
            wif_matrix = if1 - if2
            wif_contrib = wif_matrix @ effects
        return wif_contrib  # O(1) scale, same as agg_eif

    def _aggregate_overall(
        self,
        group_time_effects: Dict[Tuple[Any, Any], Dict[str, Any]],
        eif_by_gt: Dict[Tuple[Any, Any], np.ndarray],
        n_units: int,
        cohort_fractions: Dict[float, float],
        unit_cohorts: np.ndarray,
        cluster_indices: Optional[np.ndarray] = None,
        n_clusters: Optional[int] = None,
    ) -> Tuple[float, float]:
        """Compute overall ATT with WIF-adjusted SE.

        Parameters
        ----------
        group_time_effects : dict
            Group-time ATT estimates.
        eif_by_gt : dict
            Per-unit EIF values for each (g, t).
        n_units : int
            Total number of units.
        cohort_fractions : dict
            Cohort size fractions.
        unit_cohorts : ndarray, shape (n_units,)
            Cohort assignment for each unit.
        """
        # Filter to post-treatment effects
        keepers = [
            (g, t)
            for (g, t) in group_time_effects
            if t >= g - self.anticipation and np.isfinite(group_time_effects[(g, t)]["effect"])
        ]
        if not keepers:
            return np.nan, np.nan

        # Cohort-size weights
        pg = np.array([cohort_fractions.get(g, 0.0) for (g, _) in keepers])
        total_pg = pg.sum()
        if total_pg == 0:
            return np.nan, np.nan
        w = pg / total_pg

        effects = np.array([group_time_effects[gt]["effect"] for gt in keepers])
        overall_att = float(np.sum(w * effects))

        # Aggregate EIF
        agg_eif = np.zeros(n_units)
        for k, gt in enumerate(keepers):
            agg_eif += w[k] * eif_by_gt[gt]

        # WIF correction: accounts for uncertainty in cohort-size weights
        wif = self._compute_wif_contribution(
            keepers,
            effects,
            unit_cohorts,
            cohort_fractions,
            n_units,
            unit_weights=self._unit_level_weights,
        )
        # Compute SE: survey path uses score-level psi to avoid double-weighting
        # (compute_survey_vcov applies w_i internally, which would double-weight
        # the survey-weighted WIF term). Dispatch replicate vs TSL.
        if self._unit_resolved_survey is not None:
            uw = self._unit_level_weights
            # Set together with _unit_resolved_survey in fit().
            assert uw is not None
            total_w = float(np.sum(uw))
            psi_total = uw * agg_eif / total_w + wif / total_w

            if (
                hasattr(self._unit_resolved_survey, "uses_replicate_variance")
                and self._unit_resolved_survey.uses_replicate_variance
            ):
                from diff_diff.survey import compute_replicate_if_variance

                variance, _ = compute_replicate_if_variance(psi_total, self._unit_resolved_survey)
            else:
                from diff_diff.survey import compute_survey_if_variance

                variance = compute_survey_if_variance(psi_total, self._unit_resolved_survey)
            se = float(np.sqrt(max(variance, 0.0))) if np.isfinite(variance) else np.nan
        else:
            agg_eif_total = agg_eif + wif
            se = self._eif_se(agg_eif_total, n_units, cluster_indices, n_clusters)

        return overall_att, se

    def _aggregate_event_study(
        self,
        group_time_effects: Dict[Tuple[Any, Any], Dict[str, Any]],
        eif_by_gt: Dict[Tuple[Any, Any], np.ndarray],
        n_units: int,
        cohort_fractions: Dict[float, float],
        treatment_groups: List[Any],
        time_periods: List[Any],
        balance_e: Optional[int] = None,
        unit_cohorts: Optional[np.ndarray] = None,
        cluster_indices: Optional[np.ndarray] = None,
        n_clusters: Optional[int] = None,
    ) -> Dict[int, Dict[str, Any]]:
        """Aggregate ATT(g,t) by relative time e = t - g.

        Parameters
        ----------
        group_time_effects : dict
            Group-time ATT estimates.
        eif_by_gt : dict
            Per-unit EIF values for each (g, t).
        n_units : int
            Total number of units.
        cohort_fractions : dict
            Cohort size fractions.
        treatment_groups : list
            Treatment cohort identifiers.
        time_periods : list
            All time periods.
        balance_e : int, optional
            Balance event study at this relative period.
        unit_cohorts : ndarray, optional
            Cohort assignment for each unit (for WIF correction).
        """
        # Organize by relative time
        effects_by_e: Dict[int, List[Tuple[Tuple[Any, Any], float, float]]] = {}
        for (g, t), data in group_time_effects.items():
            if not np.isfinite(data["effect"]):
                continue
            e = int(t - g)
            if e not in effects_by_e:
                effects_by_e[e] = []
            effects_by_e[e].append(((g, t), data["effect"], cohort_fractions.get(g, 0.0)))

        # Balance if requested
        if balance_e is not None:
            groups_at_e = {gt[0] for gt, _, _ in effects_by_e.get(balance_e, [])}
            balanced: Dict[int, List[Tuple[Tuple[Any, Any], float, float]]] = {}
            for (g, t), data in group_time_effects.items():
                if not np.isfinite(data["effect"]):
                    continue
                if g in groups_at_e:
                    e = int(t - g)
                    if e not in balanced:
                        balanced[e] = []
                    balanced[e].append(((g, t), data["effect"], cohort_fractions.get(g, 0.0)))
            effects_by_e = balanced

        if balance_e is not None and not effects_by_e:
            warnings.warn(
                f"balance_e={balance_e}: no cohort has a finite effect at the "
                "anchor horizon. Event study will be empty.",
                UserWarning,
                stacklevel=2,
            )

        result: Dict[int, Dict[str, Any]] = {}
        for e, elist in sorted(effects_by_e.items()):
            gt_pairs = [x[0] for x in elist]
            effs = np.array([x[1] for x in elist])
            pgs = np.array([x[2] for x in elist])
            total_pg = pgs.sum()
            w = pgs / total_pg if total_pg > 0 else np.ones(len(pgs)) / len(pgs)

            agg_eff = float(np.sum(w * effs))

            # Aggregate EIF
            agg_eif = np.zeros(n_units)
            for k, gt in enumerate(gt_pairs):
                agg_eif += w[k] * eif_by_gt[gt]

            # WIF correction for event-study aggregation
            wif_e = np.zeros(n_units)
            if unit_cohorts is not None:
                es_keepers = [(g, t) for (g, t) in gt_pairs]
                es_effects = effs
                wif_e = self._compute_wif_contribution(
                    es_keepers,
                    es_effects,
                    unit_cohorts,
                    cohort_fractions,
                    n_units,
                    unit_weights=self._unit_level_weights,
                )

            if self._unit_resolved_survey is not None:
                uw = self._unit_level_weights
                # Set together with _unit_resolved_survey in fit().
                assert uw is not None
                total_w = float(np.sum(uw))
                psi_total = uw * agg_eif / total_w + wif_e / total_w

                if (
                    hasattr(self._unit_resolved_survey, "uses_replicate_variance")
                    and self._unit_resolved_survey.uses_replicate_variance
                ):
                    from diff_diff.survey import compute_replicate_if_variance

                    variance, _ = compute_replicate_if_variance(
                        psi_total, self._unit_resolved_survey
                    )
                else:
                    from diff_diff.survey import compute_survey_if_variance

                    variance = compute_survey_if_variance(psi_total, self._unit_resolved_survey)
                agg_se = float(np.sqrt(max(variance, 0.0))) if np.isfinite(variance) else np.nan
            else:
                agg_eif = agg_eif + wif_e
                agg_se = self._eif_se(agg_eif, n_units, cluster_indices, n_clusters)

            t_stat, p_val, ci = safe_inference(
                agg_eff, agg_se, alpha=self.alpha, df=self._survey_df
            )
            result[e] = {
                "effect": agg_eff,
                "se": agg_se,
                "t_stat": t_stat,
                "p_value": p_val,
                "conf_int": ci,
                "n_groups": len(elist),
            }

        return result

    def _aggregate_by_group(
        self,
        group_time_effects: Dict[Tuple[Any, Any], Dict[str, Any]],
        eif_by_gt: Dict[Tuple[Any, Any], np.ndarray],
        n_units: int,
        cohort_fractions: Dict[float, float],
        treatment_groups: List[Any],
        unit_cohorts: Optional[np.ndarray] = None,
        cluster_indices: Optional[np.ndarray] = None,
        n_clusters: Optional[int] = None,
    ) -> Dict[Any, Dict[str, Any]]:
        """Aggregate ATT(g,t) by treatment cohort.

        Parameters
        ----------
        group_time_effects : dict
            Group-time ATT estimates.
        eif_by_gt : dict
            Per-unit EIF values for each (g, t).
        n_units : int
            Total number of units.
        cohort_fractions : dict
            Cohort size fractions.
        treatment_groups : list
            Treatment cohort identifiers.
        unit_cohorts : ndarray, optional
            Cohort assignment for each unit (unused — group aggregation
            uses equal weights, not cohort-size weights).
        """
        result: Dict[Any, Dict[str, Any]] = {}
        for g in treatment_groups:
            g_gts = [
                (gg, t)
                for (gg, t) in group_time_effects
                if gg == g
                and t >= g - self.anticipation
                and np.isfinite(group_time_effects[(gg, t)]["effect"])
            ]
            if not g_gts:
                continue

            effs = np.array([group_time_effects[gt]["effect"] for gt in g_gts])
            w = np.ones(len(effs)) / len(effs)
            agg_eff = float(np.sum(w * effs))

            agg_eif = np.zeros(n_units)
            for k, gt in enumerate(g_gts):
                agg_eif += w[k] * eif_by_gt[gt]
            agg_se = self._eif_se(agg_eif, n_units, cluster_indices, n_clusters)

            t_stat, p_val, ci = safe_inference(
                agg_eff, agg_se, alpha=self.alpha, df=self._survey_df
            )
            result[g] = {
                "effect": agg_eff,
                "se": agg_se,
                "t_stat": t_stat,
                "p_value": p_val,
                "conf_int": ci,
                "n_periods": len(g_gts),
            }

        return result

    def summary(self) -> str:
        """Get summary of estimation results."""
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before calling summary()")
        assert self.results_ is not None
        return self.results_.summary()

    def print_summary(self) -> None:
        """Print summary to stdout."""
        print(self.summary())

    # -- Hausman pretest -------------------------------------------------------

    @classmethod
    def hausman_pretest(
        cls,
        data: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        first_treat: str,
        covariates: Optional[List[str]] = None,
        cluster: Optional[str] = None,
        anticipation: int = 0,
        control_group: str = "never_treated",
        alpha: float = 0.05,
        **nuisance_kwargs: Any,
    ) -> HausmanPretestResult:
        """Hausman pretest for PT-All vs PT-Post (Theorem A.1).

        Fits the estimator under both parallel trends assumptions and
        compares the results.  Under H0 (PT-All holds), both are consistent
        but PT-All is more efficient.  Rejection suggests PT-All is too
        strong; use PT-Post instead.

        Parameters
        ----------
        data, outcome, unit, time, first_treat, covariates
            Same as :meth:`fit`.
        cluster : str, optional
            Cluster column for cluster-robust covariance.
        anticipation : int
            Anticipation periods.
        control_group : str
            ``"never_treated"`` or ``"last_cohort"``.
        alpha : float
            Significance level for the test.
        **nuisance_kwargs
            Passed to both fits (e.g. ``sieve_k_max``, ``ratio_clip``,
            ``omega_ridge``).

        Returns
        -------
        HausmanPretestResult
        """
        # Fit under both assumptions (analytical SEs only, no bootstrap)
        common_kwargs = dict(
            cluster=cluster,
            control_group=control_group,
            anticipation=anticipation,
            n_bootstrap=0,
            **nuisance_kwargs,
        )
        fit_kwargs = dict(
            data=data,
            outcome=outcome,
            unit=unit,
            time=time,
            first_treat=first_treat,
            covariates=covariates,
            aggregate=None,
        )

        edid_all = cls(pt_assumption="all", alpha=alpha, **common_kwargs)
        result_all = edid_all.fit(**fit_kwargs, store_eif=True)

        edid_post = cls(pt_assumption="post", alpha=alpha, **common_kwargs)
        result_post = edid_post.fit(**fit_kwargs, store_eif=True)

        # Find common (g,t) pairs — PT-Post pairs are a subset of PT-All
        common_gts = sorted(
            set(result_all.group_time_effects.keys()) & set(result_post.group_time_effects.keys())
        )

        def _nan_result() -> HausmanPretestResult:
            return HausmanPretestResult(
                statistic=np.nan,
                p_value=np.nan,
                df=0,
                reject=False,
                alpha=alpha,
                att_all=result_all.overall_att,
                att_post=result_post.overall_att,
                recommendation="inconclusive",
                gt_details=None,
            )

        if not common_gts:
            return _nan_result()

        eif_all = result_all.influence_functions
        eif_post = result_post.influence_functions
        assert eif_all is not None and eif_post is not None
        n_units = len(next(iter(eif_all.values())))

        # --- Aggregate to post-treatment ES(e) per Theorem A.1 ---
        # Derive cohort fractions from data for proper weights
        all_units_list = sorted(data[unit].unique())
        unit_cohorts = (
            data.groupby(unit)[first_treat].first().reindex(all_units_list).values.astype(float)
        )
        cohort_fractions: Dict[float, float] = {}
        for g in set(result_all.groups) | set(result_post.groups):
            cohort_fractions[g] = float(np.sum(unit_cohorts == g)) / n_units

        def _aggregate_es(
            gt_effects: Dict, eif_dict: Dict, groups: List, ant: int
        ) -> Dict[int, Tuple[float, np.ndarray]]:
            """Aggregate (g,t) effects to post-treatment ES(e) with WIF-corrected EIF."""
            by_e: Dict[int, List[Tuple[Tuple, float, float, np.ndarray]]] = {}
            for (g, t), d in gt_effects.items():
                e = int(t - g)
                if e < -ant:
                    continue
                if not np.isfinite(d["effect"]):
                    continue
                if (g, t) not in eif_dict:
                    continue
                eif_vec = eif_dict[(g, t)]
                if not np.all(np.isfinite(eif_vec)):
                    continue
                pg = cohort_fractions.get(g, 0.0)
                if e not in by_e:
                    by_e[e] = []
                by_e[e].append(((g, t), d["effect"], pg, eif_vec))

            result: Dict[int, Tuple[float, np.ndarray]] = {}
            for e, items in by_e.items():
                if e < 0:
                    continue
                effs = np.array([x[1] for x in items])
                pgs = np.array([x[2] for x in items])
                eifs = [x[3] for x in items]
                gt_pairs_e = [x[0] for x in items]
                total_pg = pgs.sum()
                w = pgs / total_pg if total_pg > 0 else np.ones(len(pgs)) / len(pgs)
                es_eff = float(np.sum(w * effs))
                es_eif = np.zeros(n_units)
                for k_idx in range(len(eifs)):
                    es_eif += w[k_idx] * eifs[k_idx]
                # WIF correction for estimated cohort-size weights
                groups_e = np.array([g for (g, t) in gt_pairs_e])
                pg_e = np.array([cohort_fractions.get(g, 0.0) for g, t in gt_pairs_e])
                sum_pg = pg_e.sum()
                if sum_pg > 0:
                    indicator = (unit_cohorts[:, None] == groups_e[None, :]).astype(float)
                    indicator_sum = np.sum(indicator - pg_e, axis=1)
                    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                        if1 = (indicator - pg_e) / sum_pg
                        if2 = np.outer(indicator_sum, pg_e) / sum_pg**2
                        wif = (if1 - if2) @ effs
                    es_eif = es_eif + wif
                result[e] = (es_eff, es_eif)
            return result

        es_all = _aggregate_es(
            result_all.group_time_effects, eif_all, result_all.groups, anticipation
        )
        es_post = _aggregate_es(
            result_post.group_time_effects, eif_post, result_post.groups, anticipation
        )

        # Find common post-treatment horizons
        common_e = sorted(set(es_all.keys()) & set(es_post.keys()))
        if not common_e:
            return _nan_result()

        delta = np.array([es_post[e][0] - es_all[e][0] for e in common_e])

        # Build ES(e)-level EIF matrices
        eif_all_mat = np.column_stack([es_all[e][1] for e in common_e])
        eif_post_mat = np.column_stack([es_post[e][1] for e in common_e])

        # Filter units with non-finite EIF values
        row_finite = np.all(np.isfinite(eif_all_mat), axis=1) & np.all(
            np.isfinite(eif_post_mat), axis=1
        )
        cl_idx: Optional[np.ndarray] = None
        n_cl: Optional[int] = None
        if cluster is not None:
            cl_idx, n_cl = _validate_and_build_cluster_mapping(data, unit, cluster, all_units_list)
        if not np.all(row_finite):
            eif_all_mat = eif_all_mat[row_finite]
            eif_post_mat = eif_post_mat[row_finite]
            n_units = int(np.sum(row_finite))
            if cl_idx is not None:
                cl_idx = cl_idx[row_finite]
                # Recompute effective cluster count and remap to contiguous
                # indices — entire clusters may have been dropped by filtering
                unique_cl, cl_idx = np.unique(cl_idx, return_inverse=True)
                n_cl = len(unique_cl)

        # Compute full covariance matrices
        if cl_idx is not None and n_cl is not None:

            def _eif_cov(eif_mat: np.ndarray) -> np.ndarray:
                centered = _cluster_aggregate(eif_mat, cl_idx, n_cl)
                correction = n_cl / (n_cl - 1) if n_cl > 1 else 1.0
                return correction * (centered.T @ centered) / (n_units**2)

            cov_all = _eif_cov(eif_all_mat)
            cov_post = _eif_cov(eif_post_mat)
        else:
            with np.errstate(over="ignore", invalid="ignore"):
                cov_all = (eif_all_mat.T @ eif_all_mat) / (n_units**2)
                cov_post = (eif_post_mat.T @ eif_post_mat) / (n_units**2)

        H, effective_rank, p_value, n_negative, finite_ok = _hausman_quadratic_form(
            delta, cov_post, cov_all
        )
        if not finite_ok:
            warnings.warn(
                "Hausman covariance matrix contains non-finite values. " "The test is unreliable.",
                UserWarning,
                stacklevel=2,
            )
            return _nan_result()
        if n_negative > 0:
            warnings.warn(
                f"Hausman variance-difference matrix V has {n_negative} "
                "substantially negative eigenvalue(s). The test may be "
                "unreliable (finite-sample efficiency reversal).",
                UserWarning,
                stacklevel=2,
            )
        if effective_rank == 0:
            return _nan_result()
        reject = p_value < alpha

        es_details = pd.DataFrame(
            {
                "relative_period": common_e,
                "es_all": [es_all[e][0] for e in common_e],
                "es_post": [es_post[e][0] for e in common_e],
                "delta": delta,
            }
        )

        return HausmanPretestResult(
            statistic=H,
            p_value=p_value,
            df=effective_rank,
            reject=reject,
            alpha=alpha,
            att_all=result_all.overall_att,
            att_post=result_post.overall_att,
            recommendation="pt_post" if reject else "pt_all",
            gt_details=es_details,
        )
