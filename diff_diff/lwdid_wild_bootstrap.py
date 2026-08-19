"""Wild cluster bootstrap for inference with few clusters.

Thin LWDiD-facing wrapper over the house Wild Cluster Restricted (WCR)
bootstrap engine (:func:`diff_diff.utils.wild_bootstrap_se`, matched to R's
``fwildclusterboot::boottest``): the null is genuinely imposed by dropping
the treatment column from the restricted model (controls retained), the CI
is obtained by inverting the bootstrap test, and Rademacher weights are
fully enumerated automatically when ``2**G <= n_bootstrap`` and ``G <= 20``.

The wild cluster bootstrap is recommended when:

- Number of clusters G < 30
- Cluster sizes are unbalanced
- Few treated clusters

P-value convention: the house strict-exceedance count with a ~1e-9 relative
tie guard and a documented zero-p floor at ``1/(n_valid + 1)`` when that
floor is below ``alpha`` (a deliberate, documented departure from
``boottest`` — see ``diff_diff/utils.py``). This differs from the
randomization-inference module's inclusive Phipson-Smyth rule; both are
documented in ``docs/methodology/REGISTRY.md``.

References
----------
Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2008). Bootstrap-based
improvements for inference with clustered errors. *Review of Economics
and Statistics*, 90(3), 414-427.

Roodman, D., MacKinnon, J. G., Nielsen, M. O., & Webb, M. D. (2019). Fast
and wild: Bootstrap inference in Stata using boottest. *The Stata
Journal*, 19(1), 4-60.

Webb, M. D. (2014). Reworking wild bootstrap based inference for clustered
errors. *Queen's Economics Department Working Paper*, No. 1315.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from diff_diff.linalg import solve_ols
from diff_diff.utils import wild_bootstrap_se

_VALID_WEIGHT_TYPES = ("rademacher", "mammen", "webb")


@dataclass
class WildClusterBootstrapResult:
    """Result of wild cluster bootstrap inference.

    Attributes
    ----------
    att : float
        Point estimate of the average treatment effect on the treated
        (coefficient on the treatment column of the unrestricted OLS).
    se : float
        Analytical cluster-robust (CR1) standard error of ``att``. The
        studentized bootstrap drives the p-value and CI; this is not a
        rescaled bootstrap dispersion.
    t_stat_original : float
        Studentized statistic of the original estimate, ``att / se``.
    p_value : float
        Wild cluster bootstrap p-value (two-tailed; house convention —
        strict exceedance with tie guard and documented zero-p floor).
    ci_lower : float
        Lower bound of the confidence interval (by test inversion).
    ci_upper : float
        Upper bound of the confidence interval (by test inversion).
    n_clusters : int
        Number of clusters in the (post-drop) data.
    n_bootstrap : int
        Number of bootstrap replications actually performed (equals
        ``2**n_clusters`` under automatic full enumeration).
    weight_type : str
        Weight distribution used ('rademacher', 'mammen', or 'webb').
    alpha : float
        Significance level used for the CI.
    bootstrap_distribution : np.ndarray or None
        Bootstrap t* distribution (finite-filtered, so its length may be
        below ``n_bootstrap``); ``None`` when the degenerate guard fired.
    n_dropped : int
        Observations dropped for non-finite ``y`` (warned).
    """

    att: float
    se: float
    t_stat_original: float
    p_value: float
    ci_lower: float
    ci_upper: float
    n_clusters: int
    n_bootstrap: int
    weight_type: str
    alpha: float
    bootstrap_distribution: Optional[np.ndarray] = field(repr=False, default=None)
    n_dropped: int = 0

    def summary(self) -> str:
        """Return a human-readable summary string."""
        sig = (
            "***"
            if self.p_value < 0.01
            else "**" if self.p_value < 0.05 else "*" if self.p_value < 0.1 else ""
        )
        level = int(round((1 - self.alpha) * 100))
        return (
            f"Wild Cluster Bootstrap Results\n"
            f"{'=' * 50}\n"
            f"ATT: {self.att:.4f} {sig}\n"
            f"Cluster-robust (CR1) SE: {self.se:.4f}\n"
            f"{level}% CI (test inversion): [{self.ci_lower:.4f}, {self.ci_upper:.4f}]\n"
            f"P-value: {self.p_value:.4f}\n"
            f"N clusters: {self.n_clusters}\n"
            f"N bootstrap reps: {self.n_bootstrap}\n"
            f"Weight type: {self.weight_type}\n"
            f"{'=' * 50}"
        )


def wild_cluster_bootstrap(
    y: np.ndarray,
    treatment: np.ndarray,
    cluster_ids: np.ndarray,
    controls: Optional[np.ndarray] = None,
    *,
    n_bootstrap: int = 999,
    weight_type: str = "rademacher",
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> WildClusterBootstrapResult:
    """Perform wild cluster restricted bootstrap inference (CGM 2008).

    Delegates to the house engine :func:`diff_diff.utils.wild_bootstrap_se`
    (``fwildclusterboot::boottest``-matched): the null is imposed by
    re-estimating with the treatment column dropped while KEEPING the
    controls (the earlier module-local implementation fit an intercept-only
    restricted model, dumping covariate signal into the bootstrap
    residuals), the CI is obtained by test inversion, and Rademacher full
    enumeration engages automatically at ``2**G <= n_bootstrap``.

    Parameters
    ----------
    y : np.ndarray, shape (N,)
        Outcome variable. Non-finite entries are dropped with a warning
        (see ``n_dropped`` on the result).
    treatment : np.ndarray, shape (N,)
        Binary treatment indicator (0/1).
    cluster_ids : np.ndarray, shape (N,)
        Cluster membership for each observation.
    controls : np.ndarray or None, shape (N, p)
        Optional matrix of control variables. Non-finite entries raise
        ``ValueError`` (impute or remove before calling).
    n_bootstrap : int, default 999
        Number of bootstrap replications (reported as ``2**G`` when full
        enumeration engages).
    weight_type : str, default 'rademacher'
        Bootstrap weight distribution: 'rademacher', 'mammen', or 'webb'.
    alpha : float, default 0.05
        Significance level for the test-inversion confidence interval.
    seed : int or None, default None
        Random seed for reproducibility.

    Returns
    -------
    WildClusterBootstrapResult
        Point estimate with CR1 SE, test-inversion CI, bootstrap p-value,
        and the finite-filtered t* distribution.

    Raises
    ------
    ValueError
        On incompatible shapes, an invalid ``weight_type``, non-finite
        controls, or fewer than 2 clusters.

    Examples
    --------
    >>> import numpy as np
    >>> from diff_diff.lwdid_wild_bootstrap import wild_cluster_bootstrap
    >>> rng = np.random.default_rng(42)
    >>> n = 200
    >>> y = rng.normal(0, 1, n)
    >>> y[:50] += 1.5
    >>> treatment = np.zeros(n); treatment[:50] = 1.0
    >>> cluster_ids = np.repeat(np.arange(20), 10)
    >>> result = wild_cluster_bootstrap(y, treatment, cluster_ids, seed=123)
    >>> print(f"ATT={result.att:.3f}, p={result.p_value:.3f}")
    ATT=1.662, p=0.001
    """
    # ----- Input validation -----
    y = np.asarray(y, dtype=np.float64).ravel()
    treatment = np.asarray(treatment, dtype=np.float64).ravel()
    cluster_ids = np.asarray(cluster_ids).ravel()

    N = len(y)
    if N == 0:
        raise ValueError("y must not be empty.")
    if len(treatment) != N:
        raise ValueError(f"Length mismatch: y has {N} obs but treatment has {len(treatment)}.")
    if len(cluster_ids) != N:
        raise ValueError(f"Length mismatch: y has {N} obs but cluster_ids has {len(cluster_ids)}.")
    if not np.all((treatment == 0) | (treatment == 1)):
        raise ValueError(
            "treatment must be binary (0 or 1). "
            f"Got values in [{treatment.min()}, {treatment.max()}]."
        )
    if treatment.sum() == 0:
        raise ValueError("No treated observations (treatment is all zeros).")
    if treatment.sum() == N:
        raise ValueError("No control observations (treatment is all ones).")

    if controls is not None:
        controls = np.asarray(controls, dtype=np.float64)
        if controls.ndim == 1:
            controls = controls.reshape(-1, 1)
        if controls.shape[0] != N:
            raise ValueError(f"Controls have {controls.shape[0]} rows but y has {N} obs.")
        if not np.all(np.isfinite(controls)):
            raise ValueError(
                "controls contains non-finite values (NaN or Inf). "
                "Please remove or impute missing values before calling "
                "wild_cluster_bootstrap()."
            )

    if np.issubdtype(cluster_ids.dtype, np.floating) and not np.all(np.isfinite(cluster_ids)):
        raise ValueError(
            "cluster_ids contains non-finite values (NaN or Inf). "
            "Cluster identifiers must be valid for all observations."
        )
    if weight_type not in _VALID_WEIGHT_TYPES:
        raise ValueError(
            f"Unknown weight_type '{weight_type}'. Must be one of: {_VALID_WEIGHT_TYPES}"
        )
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float, np.integer, np.floating))
        or not np.isfinite(alpha)
        or not (0.0 < alpha < 1.0)
    ):
        raise ValueError(f"alpha must be a scalar in (0, 1), got {alpha!r}.")
    if (
        isinstance(n_bootstrap, bool)
        or not isinstance(n_bootstrap, (int, np.integer))
        or n_bootstrap < 2
    ):
        # Round-4 review: one draw cannot estimate a bootstrap dispersion
        # or support test inversion (matches the estimator's 0-or->=2 rule).
        raise ValueError(f"n_bootstrap must be an integer >= 2, got {n_bootstrap!r}.")

    # Drop non-finite y WITH a warning (campaign finding: silent drops).
    finite_mask = np.isfinite(y)
    n_dropped = int((~finite_mask).sum())
    if n_dropped:
        warnings.warn(
            f"wild_cluster_bootstrap: dropped {n_dropped} observation(s) "
            f"with non-finite y before estimation.",
            UserWarning,
            stacklevel=2,
        )
        y = y[finite_mask]
        treatment = treatment[finite_mask]
        cluster_ids = cluster_ids[finite_mask]
        if controls is not None:
            controls = controls[finite_mask]
        N = len(y)
        if N == 0:
            raise ValueError("All observations have non-finite y values.")
        if treatment.sum() == 0:
            raise ValueError("After dropping non-finite y, no treated observations remain.")
        if treatment.sum() == N:
            raise ValueError("After dropping non-finite y, no control observations remain.")

    unique_clusters = np.unique(cluster_ids)
    G = len(unique_clusters)
    if G < 2:
        raise ValueError(f"Need at least 2 clusters for wild cluster bootstrap, got {G}.")

    # Design matrix: [intercept, treatment, controls...]; treatment at 1.
    parts = [np.ones(N, dtype=np.float64), treatment]
    if controls is not None:
        parts.extend(controls[:, j] for j in range(controls.shape[1]))
    X = np.column_stack(parts)

    # Exactly-identified degenerate design guard (fires BEFORE delegating;
    # the shared helper stays byte-identical so its R-parity goldens cannot
    # move). With cluster-invariant treatment and G small enough that OLS
    # fits every cluster-arm mean exactly, all cluster scores are ~0 and
    # BLAS roundoff yields a tiny-positive SE instead of 0 - pre-fix this
    # reported t ~ 5e15 with p = 0.25 (below the attainable G=2 floor of
    # 0.5). Point retained; inference NaN (house fail-closed pattern).
    # Rank-aware fit through the shared solver (round-4 review: lstsq
    # returned a finite minimum-norm treatment coefficient when a control
    # duplicated the treatment column, so an unidentified ATT was reported
    # with finite bootstrap inference).
    beta_hat, resid, _ = solve_ols(X, y)
    if not np.isfinite(beta_hat[1]):
        raise ValueError(
            "The treatment coefficient is not identified: the design is "
            "rank-deficient and the shared solver dropped the treatment "
            "column (e.g. a control collinear with treatment). Remove the "
            "collinear control(s) before bootstrapping."
        )
    att_point = float(beta_hat[1])
    scores = np.array([X[cluster_ids == cl].T @ resid[cluster_ids == cl] for cl in unique_clusters])
    score_scale = float(np.abs(X.T @ np.abs(resid)).max())
    if score_scale > 0 and float(np.abs(scores).max()) <= 1e-10 * score_scale:
        warnings.warn(
            "wild_cluster_bootstrap: the cluster-level scores are exactly "
            "zero (exactly-identified design, e.g. cluster-invariant "
            "treatment with as many parameters as cluster-arm means): the "
            "cluster-robust variance is not identified. The point estimate "
            "is retained; SE, p-value, and CI are NaN.",
            UserWarning,
            stacklevel=2,
        )
        return WildClusterBootstrapResult(
            att=att_point,
            se=np.nan,
            t_stat_original=np.nan,
            p_value=np.nan,
            ci_lower=np.nan,
            ci_upper=np.nan,
            n_clusters=G,
            n_bootstrap=0,
            weight_type=weight_type,
            alpha=alpha,
            bootstrap_distribution=None,
            n_dropped=n_dropped,
        )

    house = wild_bootstrap_se(
        X,
        y,
        resid,
        cluster_ids,
        1,
        n_bootstrap=n_bootstrap,
        weight_type=weight_type,
        alpha=alpha,
        seed=seed,
        return_distribution=True,
    )

    return WildClusterBootstrapResult(
        att=att_point,
        se=float(house.se),
        t_stat_original=float(house.t_stat_original),
        p_value=float(house.p_value),
        ci_lower=float(house.ci_lower),
        ci_upper=float(house.ci_upper),
        n_clusters=int(house.n_clusters),
        n_bootstrap=int(house.n_bootstrap),
        weight_type=weight_type,
        alpha=alpha,
        bootstrap_distribution=house.bootstrap_distribution,
        n_dropped=n_dropped,
    )
