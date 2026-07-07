"""
Doubly robust math for the Efficient DiD estimator (with covariates).

Implements the with-covariates path from Chen, Sant'Anna & Xie (2025):
sieve outcome regressions (polynomial basis, AIC/BIC order selection),
sieve-based propensity score ratios (Eq 4.1-4.2), sieve-based inverse
propensities (step 4), kernel-smoothed conditional Omega*(X) for per-unit
efficient weights, doubly robust generated outcomes (Eq 4.4), and the
efficient influence function for analytical standard errors.

The DR property ensures consistency if either the outcome regression or
the sieve propensity ratio is correctly specified.  All three nuisances are
polynomial sieves / a kernel smoother (the paper's flexible-nuisance
specification, Section 4), so the doubly robust path attains the
semiparametric efficiency bound under the paper's regularity conditions
(see REGISTRY.md).

All functions are pure (no state), operating on pre-pivoted numpy arrays.
"""

import warnings
from itertools import combinations_with_replacement
from math import comb
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist

from diff_diff._backend import HAS_RUST_BACKEND, _rust_batched_ridge_chol_solve
from diff_diff.linalg import solve_ols

# Default ridge for the Omega* inversion (see ``compute_per_unit_weights`` /
# ``compute_efficient_weights``). Under PT-All the overidentified moment set
# contains telescoping near-duplicate moments, so sample Omega* is numerically
# singular (cond ~1e17-1e22 on realistic panels); the legacy pseudoinverse sits
# on the rcond-cutoff cliff, where any last-digit change to Omega* (BLAS
# reordering, platform change, 1-ulp data perturbation) moves per-cell weights
# and ATT(g,t) at the ~1e-2 relative level. The ridge
# ``Omega + lam * max(trace/H, 0) * I`` damps the statistically-null
# directions smoothly: 1-ulp per-cell stability improves from ~1e-4 to ~1e-9
# at this default, while overall-ATT bias/RMSE/coverage are unchanged
# (Monte Carlo, see REGISTRY.md). Calibrated 2026-07 against 1-ulp stability
# (target <=1e-6, achieved ~3e-9) and the HRS Table 6 anchors (all within the
# published-value tolerance at every candidate lambda).
OMEGA_RIDGE_DEFAULT = 1e-6

# ---------------------------------------------------------------------------
# Outcome regression
# ---------------------------------------------------------------------------


def estimate_outcome_regression(
    outcome_wide: np.ndarray,
    covariate_matrix: np.ndarray,
    group_mask: np.ndarray,
    t_col: int,
    tpre_col: int,
    k_max: Optional[int] = None,
    criterion: str = "bic",
    unit_weights: Optional[np.ndarray] = None,
    basis_cache: Optional[Dict[Tuple[int, int], np.ndarray]] = None,
) -> np.ndarray:
    r"""Estimate conditional mean outcome change m_hat(X) via a polynomial sieve.

    Regresses ``(Y_t - Y_{tpre})`` on a polynomial sieve basis ``psi^K(X)`` within
    the units identified by ``group_mask`` (WLS when ``unit_weights`` is given),
    selects the sieve degree ``K`` by an OLS information criterion, and returns
    predicted values ``m_hat(X_i)`` for **all** units (extrapolated from the
    within-group fit).  This implements ``m_hat_{g',t,tpre}(X) = E[Y_t - Y_{tpre}
    | G=g', X]`` as a nonparametric (sieve) regression, matching the paper's
    flexible-nuisance specification (Section 4).  Together with the sieve
    propensity ratio and kernel-smoothed Omega*(X) the doubly robust path attains
    the semiparametric efficiency bound under the paper's regularity conditions.

    Order selection mirrors :func:`estimate_propensity_ratio_sieve`.  For each
    degree ``K = 1, ..., k_max`` (capped so the basis dimension stays below the
    group size), the within-group (W)LS fit is scored by

    .. math::
        \mathrm{IC}(K) = n \, \log(\mathrm{RSS}_w / n) + c_n \, p_K

    where ``n`` is the within-group **positive-weight support** count (the raw
    row count when unweighted), ``RSS_w`` the (survey-)weighted residual sum of
    squares, ``p_K`` the basis dimension, and ``c_n = 2`` (AIC) or ``log(n)``
    (BIC).  Keying ``n`` and the penalty off the positive-weight support — only
    ``RSS_w`` is weighted — makes ``IC(K)`` shift by a ``K``-independent constant
    ``n*log(c)`` under survey-weight rescaling ``w -> c*w`` (the WLS fit itself is
    weight-scale invariant) **and** makes zero-weight rows fully inert for order
    selection, so the selected order and ``m_hat`` are invariant both to the
    survey-weight scale and to zero-weight (e.g. survey-padded) rows.

    A degree whose (weighted) design Gram matrix has condition number above
    ``1/sqrt(eps)`` (or that yields a non-finite fit) is skipped; if at least one
    degree succeeds while others are skipped a ``UserWarning`` lists them.  If
    **every** degree is skipped (e.g. the group is too small for even the linear
    basis), the estimator falls back to the intercept-only within-group mean of
    ``Y_t - Y_{tpre}`` (the unconditional outcome regression) with a
    ``UserWarning`` — distinct from the propensity sieve's constant-ratio fallback.
    An empty comparison group (``n_group == 0``) returns zeros for all units (no
    covariate adjustment).
    Degree 1 (``[1, X]``) reproduces the previous linear-OLS working model up to
    floating point.  Per-cache-miss cost rises from one OLS to up to ``k_max`` OLS
    solves, negligible against the kernel-Omega* term.

    Parameters
    ----------
    outcome_wide : ndarray, shape (n_units, n_periods)
        Pivoted outcome matrix.
    covariate_matrix : ndarray, shape (n_units, n_covariates)
        Unit-level (time-invariant) covariates.
    group_mask : ndarray of bool, shape (n_units,)
        Mask selecting units in the comparison group.
    t_col, tpre_col : int
        Column indices in ``outcome_wide`` for the two time periods.
    k_max : int or None
        Maximum polynomial degree.  None = ``floor(n_pos^{1/5})`` where ``n_pos``
        is the within-group positive-weight support count (the raw group size
        when unweighted) — a growing sieve with no fixed ceiling (the candidate
        order grows with the support size and is bounded only by
        ``n_basis < n_pos``), matching the propensity-ratio sieve.
    criterion : str
        ``"aic"`` or ``"bic"`` order selection.
    unit_weights : ndarray, shape (n_units,), optional
        Survey weights at the unit level.  When provided, uses WLS for the
        within-group regression and a weighted RSS in the criterion.

    Returns
    -------
    m_hat : ndarray, shape (n_units,)
        Predicted ``E[Y_t - Y_{tpre} | X]`` for every unit.
    """
    n_units = len(covariate_matrix)
    Y_group = outcome_wide[group_mask]
    delta_y = Y_group[:, t_col] - Y_group[:, tpre_col]
    n_group = int(np.sum(group_mask))

    if criterion not in ("aic", "bic"):
        raise ValueError(f"criterion must be 'aic' or 'bic', got {criterion!r}")

    if n_group == 0:
        return np.zeros(n_units)

    w_group = unit_weights[group_mask] if unit_weights is not None else None

    # Positive-weight support.  Zero-weight rows contribute nothing to the WLS
    # fit, the weighted Gram, or the weighted RSS, so order selection (auto-k_max,
    # the ``n_basis`` admissibility cap, and the IC sample-size terms) must key
    # off the positive-weight support count — otherwise padding the panel with
    # zero-weight (e.g. survey-subpopulation) rows could silently change the
    # selected ``K`` and hence the DR estimate even though ``m_hat`` is unchanged.
    if w_group is not None:
        support = w_group > 0
        n_pos = int(np.sum(support))
        delta_y_pos = delta_y[support]
    else:
        n_pos = n_group
        delta_y_pos = delta_y

    # Intercept-only fallback: the unconditional within-group mean of Δy.
    if w_group is not None and float(np.sum(w_group)) > 0:
        fallback_mean = float(np.average(delta_y, weights=w_group))
    else:
        fallback_mean = float(np.mean(delta_y))

    d = covariate_matrix.shape[1]
    if k_max is None:
        k_max = int(n_pos**0.2)
    k_max = max(k_max, 1)

    c_n = 2.0 if criterion == "aic" else np.log(max(n_pos, 2))
    cond_threshold = 1.0 / np.sqrt(np.finfo(float).eps)

    # Floor RSS so a (near-)perfect in-sample fit cannot drive log -> -inf and
    # spuriously select a high degree; ties then break on the K-penalty toward
    # the simpler order.
    support_var = float(np.var(delta_y_pos)) if n_pos > 0 else 0.0
    rss_floor = max(1e-300, 1e-12 * n_pos * support_var)

    best_ic = np.inf
    best_m_hat = np.full(n_units, fallback_mean)
    singular_K: List[int] = []

    for K in range(1, k_max + 1):
        n_basis = comb(K + d, d)
        # Cap so basis dimension stays below the support size (overfit guard).
        if n_basis >= n_pos:
            break

        basis_all = _sieve_basis_cached(covariate_matrix, K, basis_cache)
        basis_group = basis_all[group_mask]

        # Rank guard on the (weighted) design Gram, mirroring the propensity sieve.
        if w_group is not None:
            gram = basis_group.T @ (w_group[:, None] * basis_group)
        else:
            gram = basis_group.T @ basis_group
        with np.errstate(invalid="ignore", over="ignore"):
            gram_cond = float(np.linalg.cond(gram))
        if not np.isfinite(gram_cond) or gram_cond > cond_threshold:
            singular_K.append(K)
            continue

        coef, _, _ = solve_ols(
            basis_group,
            delta_y,
            weights=w_group,
            weight_type="pweight",
            return_vcov=False,
            rank_deficient_action="warn",
        )
        if not np.all(np.isfinite(coef)):
            singular_K.append(K)
            continue

        resid = delta_y - basis_group @ coef
        if w_group is not None:
            rss = float(np.sum(w_group * resid**2))
        else:
            rss = float(np.sum(resid**2))
        rss = max(rss, rss_floor)
        ic_val = n_pos * np.log(rss / n_pos) + c_n * n_basis

        if ic_val < best_ic:
            best_ic = ic_val
            best_m_hat = basis_all @ coef

    if best_ic == np.inf:
        warnings.warn(
            "Outcome regression sieve estimation failed for all K values "
            "(group too small or design rank-deficient at every degree). "
            "Falling back to the intercept-only within-group mean.",
            UserWarning,
            stacklevel=2,
        )
    elif singular_K:
        warnings.warn(
            f"Outcome regression sieve: skipped K={singular_K} due to "
            f"rank-deficient or non-finite design. Selected basis used the "
            f"remaining K values; this may indicate limited covariate variation.",
            UserWarning,
            stacklevel=2,
        )

    m_hat = best_m_hat
    non_finite = ~np.isfinite(m_hat)
    if non_finite.any():
        n_bad = int(non_finite.sum())
        warnings.warn(
            f"Outcome regression produced {n_bad} non-finite prediction(s). "
            "Setting to 0.0 (equivalent to no covariate adjustment).",
            UserWarning,
            stacklevel=2,
        )
        m_hat = m_hat.copy()
        m_hat[non_finite] = 0.0

    return m_hat


# ---------------------------------------------------------------------------
# Sieve-based propensity ratio estimation (Eq 4.1-4.2)
# ---------------------------------------------------------------------------


def _polynomial_sieve_basis(X: np.ndarray, degree: int) -> np.ndarray:
    """Build polynomial sieve basis up to total degree K.

    For d covariates and degree K, includes all monomials
    ``X_1^{a_1} * ... * X_d^{a_d}`` where ``a_1 + ... + a_d <= K``,
    including the intercept term (degree 0).

    Standardizes X to zero mean, unit variance for numerical stability.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Covariate matrix.
    degree : int
        Maximum total polynomial degree.

    Returns
    -------
    basis : ndarray, shape (n, n_basis)
        Sieve basis matrix. ``n_basis = C(K+d, d)``.
    """
    n, d = X.shape

    # Standardize for numerical stability (unweighted mean/std intentional —
    # this is only for conditioning, not for the statistical estimand; with
    # survey weights the sieve basis is the same, only the objective changes)
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std < 1e-10] = 1.0  # avoid division by zero for constant columns
    X_s = (X - X_mean) / X_std

    # Build monomials: enumerate all (a_1, ..., a_d) with sum <= degree
    columns = [np.ones(n)]  # degree-0 (intercept)
    for total_deg in range(1, degree + 1):
        for exponents in combinations_with_replacement(range(d), total_deg):
            col = np.ones(n)
            for idx in exponents:
                col = col * X_s[:, idx]
            columns.append(col)

    return np.column_stack(columns)


def _sieve_basis_cached(
    X: np.ndarray, degree: int, cache: Optional[Dict[Tuple[int, int], np.ndarray]]
) -> np.ndarray:
    """Per-fit memoized :func:`_polynomial_sieve_basis`.

    ``cache`` is a dict owned by one ``EfficientDiD.fit()`` and shared across the three
    sieve nuisance helpers, which all receive the same fit-level ``covariate_matrix``.
    The basis is a pure function of ``(X, degree)``, so for any degree reached by more
    than one helper (every helper starts at ``K=1`` on the same ``X``) the identical
    array would otherwise be rebuilt from scratch each time.

    Keyed on ``(id(X), degree)``: ``X`` is fixed for a fit, so the basis depends only on
    ``degree``; ``id(X)`` guards against accidental reuse of a cache with a different
    matrix. The cache lives only for the duration of one ``fit()`` (``covariate_matrix``
    stays alive throughout, so its ``id`` is stable and uncollidable), so there is no
    cross-fit leak and no ``id``-reuse hazard.

    When ``cache is None`` (the default for any standalone caller) this is a plain
    pass-through to :func:`_polynomial_sieve_basis`, leaving non-``EfficientDiD`` callers
    byte-for-byte unchanged. The helpers only read the returned array (no in-place
    mutation), so returning a shared cached object is bit-identical to rebuilding it.
    """
    if cache is None:
        return _polynomial_sieve_basis(X, degree)
    key = (id(X), degree)
    basis = cache.get(key)
    if basis is None:
        basis = _polynomial_sieve_basis(X, degree)
        cache[key] = basis
    return basis


def estimate_propensity_ratio_sieve(
    covariate_matrix: np.ndarray,
    mask_g: np.ndarray,
    mask_gp: np.ndarray,
    k_max: Optional[int] = None,
    criterion: str = "bic",
    ratio_clip: float = 20.0,
    unit_weights: Optional[np.ndarray] = None,
    basis_cache: Optional[Dict[Tuple[int, int], np.ndarray]] = None,
) -> np.ndarray:
    r"""Estimate propensity ratio via sieve convex minimization (Eq 4.1-4.2).

    Solves for each sieve degree K = 1, ..., k_max:

    .. math::
        \hat\beta_K = \arg\min_{\beta} \frac{1}{n}
            \sum_i \bigl[ G_{g',i} (\psi^K(X_i)'\beta)^2
            - 2 G_{g,i} (\psi^K(X_i)'\beta) \bigr]

    The FOC gives a closed-form linear system (no iterative optimization):
    ``(Psi_{g'}' Psi_{g'}) beta = Psi_g.sum(axis=0)``.

    Selects K via AIC/BIC: ``IC(K) = 2*loss(K) + C_n*K/n``.

    Precondition check per K: if ``cond(Psi_{g'}' W Psi_{g'}) > 1/sqrt(eps)``
    (≈ 6.7e7), that K is skipped. LinAlgError on the `np.linalg.solve` call
    or a non-finite beta skips as well. If at least one K succeeds but
    others were skipped, emits a ``UserWarning`` listing the skipped K
    values (silent-failure audit PR, axis-A finding #18). If every K is
    skipped, the caller falls back to a constant ratio of 1 with a
    separate "estimation failed for all K values" warning.

    Short-circuits ``r_{g,g}(X) = 1`` for same-cohort comparisons (PT-All).

    Parameters
    ----------
    covariate_matrix : ndarray, shape (n_units, n_covariates)
    mask_g : ndarray of bool, shape (n_units,)
        Target treatment group mask.
    mask_gp : ndarray of bool, shape (n_units,)
        Comparison group mask.
    k_max : int or None
        Maximum polynomial degree. None = ``floor(n_gp^{1/5})`` where ``n_gp`` is
        the comparison-group positive-weight support count (raw size when
        unweighted) — a growing sieve with no fixed ceiling (bounded only by
        ``n_basis < n_gp``).  Zero-weight rows do not affect order selection.
    criterion : str
        ``"aic"`` or ``"bic"``.
    ratio_clip : float
        Clip ratios to ``[1/ratio_clip, ratio_clip]``.
    unit_weights : ndarray, shape (n_units,), optional
        Survey weights at the unit level.  When provided, uses weighted
        normal equations for the sieve estimation.

    Returns
    -------
    ratio : ndarray, shape (n_units,)
        Estimated ``r_{g,g'}(X_i)`` for every unit.
    """
    n_units = len(covariate_matrix)
    n_gp = int(np.sum(mask_gp))

    # Short-circuit: r_{g,g}(X) = 1 for same-cohort comparisons (PT-All)
    if np.array_equal(mask_g, mask_gp):
        return np.ones(n_units)

    d = covariate_matrix.shape[1]

    # Survey weights and positive-weight support counts.  Zero-weight rows are
    # inert in the weighted normal equations and the weighted loss total
    # ``n_total_w``, so sieve selection (auto-k_max, the n_basis admissibility
    # cap, and the IC sample-size terms) must key off the positive-weight support
    # — otherwise padding the panel with zero-weight rows could silently change
    # the selected K and the DR estimate.  Unweighted: the raw row counts.
    if unit_weights is not None:
        w_g = unit_weights[mask_g]
        w_gp = unit_weights[mask_gp]
        n_gp_pos = int(np.sum(w_gp > 0))
        n_total_pos = int(np.sum(w_g > 0)) + n_gp_pos
        n_total_w = float(np.sum(w_g)) + float(np.sum(w_gp))
    else:
        w_g = None
        w_gp = None
        n_gp_pos = n_gp
        n_total_pos = int(np.sum(mask_g)) + n_gp
        n_total_w = float(n_total_pos)

    # Default k_max: grow with the comparison-group support size.
    if k_max is None:
        k_max = int(n_gp_pos**0.2)
    k_max = max(k_max, 1)

    # BIC penalty uses the positive-weight support count (complexity vs distinct obs)
    c_n = 2.0 if criterion == "aic" else np.log(max(n_total_pos, 2))

    best_ic = np.inf
    best_ratio = np.ones(n_units)  # fallback: constant ratio 1
    singular_K: List[int] = []  # K values skipped due to rank deficiency (#18)
    # Near-singular matrices solve without raising LinAlgError but return
    # numerically meaningless beta. Rule-of-thumb threshold: 1/sqrt(eps).
    cond_threshold = 1.0 / np.sqrt(np.finfo(float).eps)

    for K in range(1, k_max + 1):
        n_basis = comb(K + d, d)

        # Cap K so basis dimension < n_gp_pos (avoid singular system)
        if n_basis >= n_gp_pos:
            break

        basis_all = _sieve_basis_cached(covariate_matrix, K, basis_cache)
        Psi_gp = basis_all[mask_gp]  # (n_gp, n_basis)
        Psi_g = basis_all[mask_g]  # (n_g, n_basis)

        # Normal equations (weighted when survey weights present):
        # Unweighted: (Psi_gp' Psi_gp) beta = Psi_g.sum(axis=0)
        # Weighted:   (Psi_gp' W_gp Psi_gp) beta = (w_g * Psi_g).sum(axis=0)
        if w_gp is not None:
            A = Psi_gp.T @ (w_gp[:, None] * Psi_gp)
            b = (w_g[:, None] * Psi_g).sum(axis=0)
        else:
            A = Psi_gp.T @ Psi_gp
            b = Psi_g.sum(axis=0)

        # Precondition check (#18, axis A): reject near-singular A explicitly
        # so np.linalg.solve can't silently return garbage coefficients.
        with np.errstate(invalid="ignore", over="ignore"):
            A_cond = float(np.linalg.cond(A))
        if not np.isfinite(A_cond) or A_cond > cond_threshold:
            singular_K.append(K)
            continue

        try:
            beta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            singular_K.append(K)
            continue  # singular — try next K

        # Check for NaN/Inf in solution
        if not np.all(np.isfinite(beta)):
            singular_K.append(K)
            continue

        # Predicted ratio for all units
        r_hat = basis_all @ beta

        # IC selection: loss at optimum = -(1/n_w) * b'beta
        # Derivation: L(beta) = (1/n_w)(beta'A*beta - 2*b'beta).
        # At optimum A*beta = b, so beta'A*beta = b'beta.
        # Therefore L = (1/n_w)(b'beta - 2*b'beta) = -(1/n_w)*b'beta.
        # Loss uses weighted totals; BIC penalty uses the positive-weight support.
        loss = -float(b @ beta) / n_total_w
        ic_val = 2.0 * loss + c_n * n_basis / n_total_pos

        if ic_val < best_ic:
            best_ic = ic_val
            best_ratio = r_hat.copy()

    # Warn if no sieve fit succeeded (falling back to constant ratio 1)
    if best_ic == np.inf:
        warnings.warn(
            "Propensity ratio sieve estimation failed for all K values. "
            "Falling back to constant ratio of 1 (no ratio adjustment). "
            "The DR estimator relies on outcome regression only.",
            UserWarning,
            stacklevel=2,
        )
    elif singular_K:
        # Finding #18 (axis A): partial K-failure was previously silent.
        # Surface it so users see that the selected basis order was
        # forced by rank deficiency at higher K rather than by the IC.
        warnings.warn(
            f"Propensity ratio sieve: skipped K={singular_K} due to "
            f"rank-deficient or non-finite normal equations. "
            f"Selected basis used the remaining K values; "
            f"this may indicate limited variation in the covariates.",
            UserWarning,
            stacklevel=2,
        )

    # Overlap diagnostics: warn if ratios require significant clipping
    n_extreme = int(np.sum((best_ratio < 1.0 / ratio_clip) | (best_ratio > ratio_clip)))
    if n_extreme > 0:
        pct = 100.0 * n_extreme / n_units
        warnings.warn(
            f"Sieve propensity ratios for {n_extreme} of {n_units} units "
            f"({pct:.1f}%) were outside [{1.0/ratio_clip:.2f}, {ratio_clip:.1f}] "
            f"and will be clipped. This may indicate overlap assumption "
            f"violations (near-zero propensity scores for some covariate values).",
            UserWarning,
            stacklevel=2,
        )

    # Clip: population ratio p_g(X)/p_{g'}(X) is non-negative
    best_ratio = np.clip(best_ratio, 1.0 / ratio_clip, ratio_clip)

    return best_ratio


# ---------------------------------------------------------------------------
# Sieve-based inverse propensity estimation (Algorithm step 4)
# ---------------------------------------------------------------------------


def estimate_inverse_propensity_sieve(
    covariate_matrix: np.ndarray,
    group_mask: np.ndarray,
    k_max: Optional[int] = None,
    criterion: str = "bic",
    unit_weights: Optional[np.ndarray] = None,
    basis_cache: Optional[Dict[Tuple[int, int], np.ndarray]] = None,
) -> np.ndarray:
    r"""Estimate s_{g'}(X) = 1/p_{g'}(X) via sieve convex minimization.

    Solves for each sieve degree K:

    .. math::
        \hat\beta_K = \arg\min_\beta \frac{1}{n}
            \sum_i \bigl[ G_{g',i} (\psi^K(X_i)'\beta)^2
            - 2 (\psi^K(X_i)'\beta) \bigr]

    FOC: ``(Psi_{g'}' Psi_{g'}) beta = Psi_all.sum(axis=0)``

    This is the same structure as the ratio estimator but with all
    units on the RHS (not just group g), following the paper's
    algorithm step 4.

    Precondition check per K: if ``cond(Psi_{g'}' W Psi_{g'}) > 1/sqrt(eps)``
    (≈ 6.7e7), that K is skipped. LinAlgError on the `np.linalg.solve` call
    or a non-finite beta skips as well. If at least one K succeeds but
    others were skipped, emits a ``UserWarning`` listing the skipped K
    values (silent-failure audit PR, axis-A finding #18). If every K is
    skipped, the caller falls back to unconditional ``n/n_group`` scaling
    with a separate "estimation failed for all K values" warning.

    Parameters
    ----------
    covariate_matrix : ndarray, shape (n_units, n_covariates)
    group_mask : ndarray of bool, shape (n_units,)
        Mask for the group whose inverse propensity to estimate.
    k_max : int or None
        Maximum polynomial degree. None = auto.
    criterion : str
        ``"aic"`` or ``"bic"``.
    unit_weights : ndarray, shape (n_units,), optional
        Survey weights at the unit level.  When provided, uses weighted
        normal equations for the sieve estimation.

    Returns
    -------
    s_hat : ndarray, shape (n_units,)
        Estimated ``1/p_{g'}(X_i)`` for every unit. Clipped to [1, n].
    """
    n_units = len(covariate_matrix)
    n_group = int(np.sum(group_mask))
    d = covariate_matrix.shape[1]

    if n_group == 0:
        return np.ones(n_units)

    # Survey weights, fallback, and positive-weight support counts.  Zero-weight
    # rows are inert in the weighted normal equations and the weighted loss total
    # ``n_units_w``, so sieve selection (auto-k_max, the n_basis admissibility
    # cap, and the IC sample-size terms) must key off the positive-weight support
    # rather than the raw row counts (see the outcome-regression docstring) —
    # padding with zero-weight rows then cannot change the selected K or the DR
    # estimate.  Unweighted: the raw row counts.
    if unit_weights is not None:
        w_group = unit_weights[group_mask]
        sum_w_group = float(np.sum(w_group))
        if sum_w_group <= 0:
            # Zero survey weight for this group — return unconditional fallback
            return np.ones(n_units)
        n_units_w = float(np.sum(unit_weights))
        fallback_ratio = n_units_w / sum_w_group
        n_group_pos = int(np.sum(w_group > 0))
        n_units_pos = int(np.sum(unit_weights > 0))
    else:
        w_group = None
        n_units_w = float(n_units)
        fallback_ratio = n_units / n_group
        n_group_pos = n_group
        n_units_pos = n_units

    if k_max is None:
        k_max = int(n_group_pos**0.2)
    k_max = max(k_max, 1)

    # BIC penalty uses the positive-weight support count
    c_n = 2.0 if criterion == "aic" else np.log(max(n_units_pos, 2))

    best_ic = np.inf
    best_s = np.full(n_units, fallback_ratio)  # fallback: unconditional
    singular_K: List[int] = []  # K values skipped due to rank deficiency (#18)
    cond_threshold = 1.0 / np.sqrt(np.finfo(float).eps)

    for K in range(1, k_max + 1):
        n_basis = comb(K + d, d)
        if n_basis >= n_group_pos:
            break

        basis_all = _sieve_basis_cached(covariate_matrix, K, basis_cache)
        Psi_gp = basis_all[group_mask]

        # Normal equations (weighted when survey weights present):
        # Unweighted: (Psi_gp' Psi_gp) beta = Psi_all.sum(axis=0)
        # Weighted:   (Psi_gp' W_group Psi_gp) beta = (w_all * Psi_all).sum(axis=0)
        if w_group is not None:
            A = Psi_gp.T @ (w_group[:, None] * Psi_gp)
            b = (unit_weights[:, None] * basis_all).sum(axis=0)
        else:
            A = Psi_gp.T @ Psi_gp
            # RHS: sum of basis over ALL units (not just one group)
            b = basis_all.sum(axis=0)

        # Precondition check (#18, axis A): see ratio-sieve comment above.
        with np.errstate(invalid="ignore", over="ignore"):
            A_cond = float(np.linalg.cond(A))
        if not np.isfinite(A_cond) or A_cond > cond_threshold:
            singular_K.append(K)
            continue

        try:
            beta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            singular_K.append(K)
            continue
        if not np.all(np.isfinite(beta)):
            singular_K.append(K)
            continue

        s_hat = basis_all @ beta

        # IC: loss = -(1/n_w) * b'beta (same derivation as ratio estimator)
        # Loss uses weighted totals; BIC penalty uses the positive-weight support.
        loss = -float(b @ beta) / n_units_w
        ic_val = 2.0 * loss + c_n * n_basis / n_units_pos

        if ic_val < best_ic:
            best_ic = ic_val
            best_s = s_hat.copy()

    # Warn if no sieve fit succeeded (falling back to unconditional)
    if best_ic == np.inf:
        warnings.warn(
            "Inverse propensity sieve estimation failed for all K values. "
            "Falling back to unconditional n/n_group scaling.",
            UserWarning,
            stacklevel=2,
        )
    elif singular_K:
        # Finding #18 (axis A): partial K-failure was previously silent.
        warnings.warn(
            f"Inverse propensity sieve: skipped K={singular_K} due to "
            f"rank-deficient or non-finite normal equations. "
            f"Selected basis used the remaining K values; "
            f"this may indicate limited variation in the covariates.",
            UserWarning,
            stacklevel=2,
        )

    # Overlap diagnostics: warn if s_hat values require clipping
    n_clipped = int(np.sum((best_s < 1.0) | (best_s > float(n_units))))
    if n_clipped > 0:
        pct = 100.0 * n_clipped / n_units
        warnings.warn(
            f"Inverse propensity estimates for {n_clipped} of {n_units} units "
            f"({pct:.1f}%) were outside [1, {n_units}] and will be clipped. "
            f"This may indicate overlap assumption violations.",
            UserWarning,
            stacklevel=2,
        )

    # s = 1/p must be >= 1 (since p <= 1) and bounded above
    best_s = np.clip(best_s, 1.0, float(n_units))
    return best_s


# ---------------------------------------------------------------------------
# Doubly robust generated outcomes (Eq 4.4)
# ---------------------------------------------------------------------------


def compute_generated_outcomes_cov(
    target_g: float,
    target_t: float,
    valid_pairs: List[Tuple[float, float]],
    outcome_wide: np.ndarray,
    cohort_masks: Dict[float, np.ndarray],
    never_treated_mask: np.ndarray,
    period_to_col: Dict[float, int],
    period_1_col: int,
    cohort_fractions: Dict[float, float],
    m_hat_cache: Dict[Tuple, np.ndarray],
    r_hat_cache: Dict[Tuple[float, float], np.ndarray],
    never_treated_val: float = np.inf,
) -> np.ndarray:
    """Compute per-unit doubly robust generated outcomes (Eq 4.4).

    For each valid pair ``(g', t_pre)`` and each unit ``i``, three terms::

        Term 1 (treated):
            (G_{g,i} / pi_g) * (Y_{i,t} - Y_{i,1}
                - m_{inf,t,tpre}(X_i) - m_{g',tpre,1}(X_i))

        Term 2 (never-treated):
            -r_{g,inf}(X_i) * (G_{inf,i} / pi_g)
                * (Y_{i,t} - Y_{i,tpre} - m_{inf,t,tpre}(X_i))

        Term 3 (comparison cohort):
            -r_{g,g'}(X_i) * (G_{g',i} / pi_g)
                * (Y_{i,tpre} - Y_{i,1} - m_{g',tpre,1}(X_i))

    Returns
    -------
    gen_out : ndarray, shape (n_units, H)
        Per-unit generated outcome for each valid pair.
    """
    H = len(valid_pairs)
    n_units = outcome_wide.shape[0]
    if H == 0:
        return np.empty((n_units, 0))

    t_col = period_to_col[target_t]
    y1_col = period_1_col

    g_mask = cohort_masks[target_g]
    pi_g = cohort_fractions[target_g]

    # Guard: zero survey weight for the target cohort → no DR estimation possible
    if pi_g <= 0:
        return np.zeros((n_units, H))

    gen_out = np.zeros((n_units, H))

    for j, (gp, tpre) in enumerate(valid_pairs):
        tpre_col = period_to_col[tpre]

        m_inf_t_tpre = m_hat_cache[(never_treated_val, t_col, tpre_col)]
        m_gp_tpre_1 = m_hat_cache[(gp, tpre_col, y1_col)]
        r_g_inf = r_hat_cache[(target_g, never_treated_val)]
        r_g_gp = r_hat_cache[(target_g, gp)]

        # Term 1: treated units
        if pi_g > 0:
            Y_t_minus_Y1 = outcome_wide[g_mask, t_col] - outcome_wide[g_mask, y1_col]
            residual_treated = Y_t_minus_Y1 - m_inf_t_tpre[g_mask] - m_gp_tpre_1[g_mask]
            gen_out[g_mask, j] += (1.0 / pi_g) * residual_treated

        # Term 2: never-treated units
        pi_inf = cohort_fractions.get(never_treated_val, 0.0)
        if pi_inf > 0:
            Y_t_minus_Ytpre = (
                outcome_wide[never_treated_mask, t_col] - outcome_wide[never_treated_mask, tpre_col]
            )
            residual_inf = Y_t_minus_Ytpre - m_inf_t_tpre[never_treated_mask]
            gen_out[never_treated_mask, j] -= (
                r_g_inf[never_treated_mask] * (1.0 / pi_g) * residual_inf
            )

        # Term 3: comparison cohort units
        if np.isinf(gp):
            gp_mask = never_treated_mask
        else:
            gp_mask = cohort_masks[gp]
        pi_gp = cohort_fractions.get(gp, 0.0)
        if pi_gp > 0:
            Y_tpre_minus_Y1 = outcome_wide[gp_mask, tpre_col] - outcome_wide[gp_mask, y1_col]
            residual_gp = Y_tpre_minus_Y1 - m_gp_tpre_1[gp_mask]
            gen_out[gp_mask, j] -= r_g_gp[gp_mask] * (1.0 / pi_g) * residual_gp

    return gen_out


# ---------------------------------------------------------------------------
# Kernel-smoothed conditional Omega* (Eq 3.12)
# ---------------------------------------------------------------------------


def _silverman_bandwidth(X: np.ndarray, unit_weights: Optional[np.ndarray] = None) -> float:
    """Silverman's rule-of-thumb bandwidth for d-dimensional X.

    ``h = (4 / (d + 2))^{1/(d+4)} * median_std * n^{-1/(d+4)}``

    When ``unit_weights`` is provided, the rule is evaluated on the
    **positive-weight support** (rows with ``w > 0``) only, and the per-dimension
    dispersion is **survey-weighted**: ``median_std`` is the median across
    covariate dimensions of the weighted standard deviation
    ``sqrt(sum_i w_i (x_i - xbar_w)^2 / sum_i w_i)`` with the weighted mean
    ``xbar_w = sum_i w_i x_i / sum_i w_i``. Survey-weighted moments reflect the
    population distribution the kernel-smoothed ``Omega*(X)`` targets, rather than
    the unweighted sample. The rate term ``n`` remains the positive-weight support
    count (the dispersion is weighted; the sample-size term is not — a deliberately
    scoped refinement, not Kish ``n_eff``).

    Invariances preserved:

    - **Weight scale** (``w -> c*w``, ``c > 0``): the weighted mean/std and the
      positive-weight count are all invariant, so the bandwidth is unchanged.
    - **Zero-weight (survey-subpopulation / padded) rows**: zero-weight rows drop
      from the support, contribute nothing to the weighted moments, and do not
      change the count, so the bandwidth — and hence ``Omega*(X)`` and the
      per-unit efficient weights it feeds — is invariant to such padding (e.g. a
      zero-weight row with an extreme covariate cannot inflate ``median_std``).
    - **Uniform positive weights**: the weighted std reduces to the unweighted
      population std, matching the pre-refinement bandwidth up to floating point.

    Falls back to the unweighted full matrix when no weights are given or the
    positive-weight support is empty.
    """
    weights = None
    if unit_weights is not None:
        support = unit_weights > 0
        if np.any(support):
            X = X[support]
            weights = unit_weights[support]
    n, d = X.shape
    if weights is not None:
        w_norm = weights / weights.sum()
        weighted_mean = w_norm @ X
        weighted_var = w_norm @ (X - weighted_mean) ** 2
        stds = np.sqrt(weighted_var)
    else:
        stds = np.std(X, axis=0)
    stds[stds < 1e-10] = 1.0
    median_std = float(np.median(stds))
    h = (4.0 / (d + 2)) ** (1.0 / (d + 4)) * median_std * n ** (-1.0 / (d + 4))
    return max(h, 1e-10)


def _kernel_weights_matrix(
    X_all: np.ndarray,
    X_group: np.ndarray,
    bandwidth: float,
    group_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Gaussian kernel weight matrix.

    Returns shape ``(n_all, n_group)`` where entry ``[i, j]`` is the
    normalized kernel weight ``K_h(X_group[j], X_all[i])``.

    Each row sums to 1 (Nadaraya-Watson normalization).

    Parameters
    ----------
    group_weights : ndarray, shape (n_group,), optional
        Survey weights for the group units.  When provided, kernel
        weights are multiplied by survey weights before row-normalization,
        making the Nadaraya-Watson estimator survey-weighted.
    """
    # Squared distances: (n_all, n_group)
    dist_sq = cdist(X_all, X_group, metric="sqeuclidean")
    # Gaussian kernel
    raw = np.exp(-dist_sq / (2.0 * bandwidth**2))
    # Survey-weight: each group unit j contributes ∝ w_j * K_h(X_i, X_j)
    if group_weights is not None:
        raw = raw * group_weights[np.newaxis, :]
    # Normalize each row
    row_sums = raw.sum(axis=1, keepdims=True)
    row_sums[row_sums < 1e-15] = 1.0  # avoid division by zero
    return raw / row_sums


def _kernel_weighted_cov(
    A: np.ndarray,
    B: np.ndarray,
    W: np.ndarray,
) -> np.ndarray:
    """Kernel-weighted local covariance.

    Parameters
    ----------
    A : ndarray, shape (n_group,)
    B : ndarray, shape (n_group,)
    W : ndarray, shape (n_all, n_group)
        Normalized kernel weights (rows sum to 1).

    Returns
    -------
    cov : ndarray, shape (n_all,)
        ``Cov_hat(A, B | X_i)`` for each target unit i.
    """
    # Local means: (n_all,)
    A_local = W @ A
    B_local = W @ B

    # Centered products: (n_all, n_group)
    A_centered = A[np.newaxis, :] - A_local[:, np.newaxis]  # (n_all, n_group)
    B_centered = B[np.newaxis, :] - B_local[:, np.newaxis]

    # Weighted local covariance: (n_all,)
    cov = np.sum(W * A_centered * B_centered, axis=1)
    return cov


def compute_omega_star_conditional(
    target_g: float,
    target_t: float,
    valid_pairs: List[Tuple[float, float]],
    outcome_wide: np.ndarray,
    cohort_masks: Dict[float, np.ndarray],
    never_treated_mask: np.ndarray,
    period_to_col: Dict[float, int],
    period_1_col: int,
    cohort_fractions: Dict[float, float],
    covariate_matrix: np.ndarray,
    s_hat_cache: Dict[float, np.ndarray],
    bandwidth: Optional[float] = None,
    unit_weights: Optional[np.ndarray] = None,
    never_treated_val: float = np.inf,
) -> np.ndarray:
    r"""Kernel-smoothed conditional Omega\*(X_i) for each unit (Eq 3.12).

    Estimates the five-term conditional covariance matrix using
    Nadaraya-Watson kernel regression with Gaussian kernel and
    local (kernel-weighted) means.  Scales each term by per-unit
    conditional inverse propensities ``s_hat_g(X_i) = 1/p_g(X_i)``
    (algorithm step 4), matching the paper's Eq 3.12.

    Parameters
    ----------
    target_g, target_t : float
        Target group-time.
    valid_pairs : list of (g', t_pre)
    outcome_wide : ndarray, shape (n_units, n_periods)
    cohort_masks, never_treated_mask, period_to_col, period_1_col,
    cohort_fractions : pre-computed data structures
    covariate_matrix : ndarray, shape (n_units, n_covariates)
    s_hat_cache : dict
        Inverse propensity estimates ``{group: s_hat(X_i)}`` where each
        value is shape ``(n_units,)``. Keyed by group identifier.
    bandwidth : float or None
        Kernel bandwidth. None = Silverman's rule.
    unit_weights : ndarray, shape (n_units,), optional
        Survey weights at the unit level.  When provided, kernel-smoothed
        covariances use survey-weighted Nadaraya-Watson regression.
    never_treated_val : float

    Returns
    -------
    omega : ndarray, shape (n_units, H, H)
        Per-unit conditional covariance matrices.
    """
    H = len(valid_pairs)
    n_units = outcome_wide.shape[0]
    if H == 0:
        return np.empty((n_units, 0, 0))

    if bandwidth is None:
        bandwidth = _silverman_bandwidth(covariate_matrix, unit_weights)

    t_col = period_to_col[target_t]
    y1_col = period_1_col

    g_mask = cohort_masks[target_g]

    Y_inf = outcome_wide[never_treated_mask]
    X_inf = covariate_matrix[never_treated_mask]

    # Per-unit inverse propensities from sieve estimation (Eq 3.12)
    s_g = s_hat_cache.get(target_g, np.full(n_units, 1.0 / max(cohort_fractions[target_g], 1e-10)))
    s_inf = s_hat_cache.get(
        never_treated_val,
        np.full(n_units, 1.0 / max(cohort_fractions.get(never_treated_val, 1e-10), 1e-10)),
    )

    # Scalability warning
    if n_units > 5000:
        warnings.warn(
            f"Conditional Omega* estimation with n={n_units} is expensive "
            f"(O(n^2 * H^2)). Consider using fewer units.",
            UserWarning,
            stacklevel=2,
        )

    # Per-group survey weights for kernel smoothing
    w_g = unit_weights[g_mask] if unit_weights is not None else None
    w_inf = unit_weights[never_treated_mask] if unit_weights is not None else None

    # Pre-compute kernel weight matrices per group
    Y_g = outcome_wide[g_mask]
    X_g = covariate_matrix[g_mask]
    Yg_t_minus_1 = Y_g[:, t_col] - Y_g[:, y1_col]

    W_g = _kernel_weights_matrix(covariate_matrix, X_g, bandwidth, group_weights=w_g)
    W_inf = _kernel_weights_matrix(covariate_matrix, X_inf, bandwidth, group_weights=w_inf)

    inf_t_minus_tpre = {}
    for _, tpre in valid_pairs:
        tpre_col = period_to_col[tpre]
        if tpre_col not in inf_t_minus_tpre:
            inf_t_minus_tpre[tpre_col] = Y_inf[:, t_col] - Y_inf[:, tpre_col]

    W_gp_cache: Dict[float, np.ndarray] = {}
    gp_outcomes_cache: Dict[float, np.ndarray] = {}

    omega = np.zeros((n_units, H, H))

    # Term 1: s_g(X) * Cov(Y_t-Y_1, Y_t-Y_1 | G=g, X) — same for all (j,k)
    term1 = s_g * _kernel_weighted_cov(Yg_t_minus_1, Yg_t_minus_1, W_g)

    for j in range(H):
        gp_j, tpre_j = valid_pairs[j]
        tpre_j_col = period_to_col[tpre_j]

        for k in range(j, H):
            gp_k, tpre_k = valid_pairs[k]
            tpre_k_col = period_to_col[tpre_k]

            val = term1.copy()

            # Term 2: s_inf(X) * Cov(Y_t-Y_{tpre_j}, Y_t-Y_{tpre_k} | G=inf, X)
            val += s_inf * _kernel_weighted_cov(
                inf_t_minus_tpre[tpre_j_col],
                inf_t_minus_tpre[tpre_k_col],
                W_inf,
            )

            # Term 3: -1{g==g'_j} * s_g(X) * Cov(Y_t-Y_1, Y_{tpre_j}-Y_1 | G=g, X)
            if gp_j == target_g:
                g_tpre_j = Y_g[:, tpre_j_col] - Y_g[:, y1_col]
                val -= s_g * _kernel_weighted_cov(Yg_t_minus_1, g_tpre_j, W_g)

            # Term 4: -1{g==g'_k} * s_g(X) * Cov(Y_t-Y_1, Y_{tpre_k}-Y_1 | G=g, X)
            if gp_k == target_g:
                g_tpre_k = Y_g[:, tpre_k_col] - Y_g[:, y1_col]
                val -= s_g * _kernel_weighted_cov(Yg_t_minus_1, g_tpre_k, W_g)

            # Term 5: 1{g'_j==g'_k} * s_{g'_j}(X) * Cov(...)
            if gp_j == gp_k:
                if np.isinf(gp_j):
                    inf_tpre_j = Y_inf[:, tpre_j_col] - Y_inf[:, y1_col]
                    inf_tpre_k = Y_inf[:, tpre_k_col] - Y_inf[:, y1_col]
                    val += s_inf * _kernel_weighted_cov(inf_tpre_j, inf_tpre_k, W_inf)
                else:
                    s_gp_j = s_hat_cache.get(
                        gp_j, np.full(n_units, 1.0 / max(cohort_fractions.get(gp_j, 1e-10), 1e-10))
                    )
                    if gp_j not in W_gp_cache:
                        X_gp = covariate_matrix[cohort_masks[gp_j]]
                        w_gp_j = (
                            unit_weights[cohort_masks[gp_j]] if unit_weights is not None else None
                        )
                        W_gp_cache[gp_j] = _kernel_weights_matrix(
                            covariate_matrix, X_gp, bandwidth, group_weights=w_gp_j
                        )
                        gp_outcomes_cache[gp_j] = outcome_wide[cohort_masks[gp_j]]
                    W_gp = W_gp_cache[gp_j]
                    Y_gp = gp_outcomes_cache[gp_j]
                    gp_tpre_j = Y_gp[:, tpre_j_col] - Y_gp[:, y1_col]
                    gp_tpre_k = Y_gp[:, tpre_k_col] - Y_gp[:, y1_col]
                    val += s_gp_j * _kernel_weighted_cov(gp_tpre_j, gp_tpre_k, W_gp)

            omega[:, j, k] = val
            if j != k:
                omega[:, k, j] = val

    return omega


# ---------------------------------------------------------------------------
# Fused tiled GEMM path: conditional Omega* + ridge weights + DR scores
# ---------------------------------------------------------------------------

# Memory cap for the unit-tiled conditional path. Resolved at CALL time inside
# compute_conditional_cells_tiled (never bound as a def-time default) so tests
# can monkeypatch it to force multi-tile execution.
_TARGET_OMEGA_TILE_BYTES = 256 * 1024 * 1024

# Column-chunk width for the product GEMMs inside _build_group_kcov_table.
# Bounds the (n_group x chunk) product temporary independently of how many
# distinct kernel-covariance columns a fit needs.
_KCOV_PRODUCT_CHUNK = 512


def _kcov_batch(W: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Batched kernel-weighted local covariances via GEMM.

    For row-normalized kernel weights ``W`` (each row sums to 1),

        KCov_i(A, B) = sum_j W_ij (A_j - Abar_i)(B_j - Bbar_i)
                     = (W @ (A0 * B0))_i - (W @ A0)_i * (W @ B0)_i

    with ``A0 = A - mean(A)`` (global pre-centering keeps the
    uncentered-product form cancellation-safe; any constant shift cancels in
    the identity). Row normalization is per-row, so the identity holds for
    survey-weighted kernels and for any row-tile of ``W``.

    Parameters
    ----------
    W : ndarray, shape (n_rows, n_group)
        Row-normalized kernel weight matrix (or a row-tile of one).
    A, B : ndarray, shape (n_group, m)
        Column-matched inputs; column ``j`` of the output is
        ``KCov_i(A[:, j], B[:, j])``.

    Returns
    -------
    ndarray, shape (n_rows, m)
    """
    A0 = A - A.mean(axis=0)
    B0 = B - B.mean(axis=0)
    return W @ (A0 * B0) - (W @ A0) * (W @ B0)


def _collect_kcov_specs(
    cell_specs: List[Dict],
    period_to_col: Dict[float, int],
    never_treated_val: float,
) -> Tuple[Dict[float, Dict], Dict[Tuple[float, float], Dict]]:
    """Cross-cell kernel-covariance key registry + per-cell gather specs.

    Every Omega* entry is a sum of ``s_group * KCov(Y_u1 - Y_v1, Y_u2 - Y_v2
    | group)`` terms, keyed only by wide-outcome column indices - not by the
    (g, t) cell - so the H(H+1)/2-pair tables that
    ``compute_conditional_cells_tiled`` used to rebuild per cell dedup to one
    table of distinct product columns per comparison group per tile.

    Difference keys are ORDERED ``(minuend_col, subtrahend_col)`` and are
    never sort-canonicalized: ``Y_u - Y_v = -(Y_v - Y_u)``, so sorting would
    sign-flip covariances for cells whose pre-periods straddle t (possible
    under PT-All).  Only the symmetric product-pair swap is canonicalized
    (``KCov(A, B) = KCov(B, A)``).

    Per-cell gather specs cover the UPPER TRIANGLE only; assembly builds a
    ``(tile, H(H+1)/2)`` strip and writes it to both triangles, reproducing
    the legacy loop's exact ``omega[k, j] = omega[j, k]`` mirror semantics
    (computing the lower triangle independently would reassociate the
    term-3/4 subtractions).  Gathers go through compact per-cell column
    slices of the group tables (cache-resident, ~3x faster than gathering
    from the full table).  Masked-out entries (no same-cohort term 3/4, no
    matching-group term 5) resolve to an all-zero column: ``x - 0.0 == x``
    and ``x + 0.0 == x`` value-exactly in IEEE, matching the legacy
    conditional skips.

    Returns
    -------
    registry : dict mapping group key to
        ``{"diff_keys": (nD, 2) int64 array, "prod_i", "prod_j": (nP,) int64
        arrays (indices into diff_keys), "zero_col": nP}``.
        Only groups needed by some H >= 2 cell appear; H == 1 cells skip
        Omega* entirely and contribute no keys (so e.g. a pure
        pt_assumption="post" fit builds no tables and no kernel matrices).
    cell_asm : dict mapping (g, t) of each H >= 2 cell to triu-entry specs
        (entry order = ``np.triu_indices(H)``):
        ``H``, ``iu``/``ju`` (triu index vectors), ``t1_col`` (group-g table
        column), ``t2_cols``/``t2_loc`` (compact never-treated-table columns
        + per-entry local map), ``t34_cols``/``t34_row_loc``/``t34_col_loc``
        (compact group-g columns incl. its zero column, or ``t34_cols =
        None`` when the cell has no same-cohort pair), ``t5_parts`` (list of
        (gkey, compact column array) in slot order), ``t5_loc`` (per-entry
        local map into the slot-concatenated layout whose last column is
        zero), ``m5`` (total concat width excl. the zero column).
    """
    diff_index: Dict[float, Dict[Tuple[int, int], int]] = {}
    prod_index: Dict[float, Dict[Tuple[int, int], int]] = {}

    def _prod_col(gkey: float, d1: Tuple[int, int], d2: Tuple[int, int]) -> int:
        dmap = diff_index.setdefault(gkey, {})
        pmap = prod_index.setdefault(gkey, {})
        i1 = dmap.setdefault(d1, len(dmap))
        i2 = dmap.setdefault(d2, len(dmap))
        pkey = (i1, i2) if i1 <= i2 else (i2, i1)
        return pmap.setdefault(pkey, len(pmap))

    cell_asm: Dict[Tuple[float, float], Dict] = {}
    for spec in cell_specs:
        pairs = spec["pairs"]
        H = len(pairs)
        if H < 2:
            continue
        g = spec["g"]
        t_col = spec["t_col"]
        y1_col = spec["y1_col"]
        j_c = [period_to_col[tp] for _, tp in pairs]
        j_gp = [gp for gp, _ in pairs]
        j_gkey = [never_treated_val if np.isinf(gp) else gp for gp in j_gp]

        t1_col = _prod_col(g, (t_col, y1_col), (t_col, y1_col))
        has_t34 = any(gp == g for gp in j_gp)
        iu, ju = np.triu_indices(H)
        n_tri = iu.size
        t2_ids = np.empty(n_tri, dtype=np.int64)
        t34_row_ids = np.full(n_tri, -1, dtype=np.int64)
        t34_col_ids = np.full(n_tri, -1, dtype=np.int64)
        cell_gkeys = sorted(set(j_gkey))
        slot_of = {gk: s for s, gk in enumerate(cell_gkeys)}
        t5_slot = np.full(n_tri, -1, dtype=np.int64)
        t5_ids = np.full(n_tri, -1, dtype=np.int64)

        for e in range(n_tri):
            j = int(iu[e])
            k = int(ju[e])
            cj, ck = j_c[j], j_c[k]
            t2_ids[e] = _prod_col(never_treated_val, (t_col, cj), (t_col, ck))
            if has_t34:
                if j_gp[j] == g:
                    t34_row_ids[e] = _prod_col(g, (t_col, y1_col), (cj, y1_col))
                if j_gp[k] == g:
                    t34_col_ids[e] = _prod_col(g, (t_col, y1_col), (ck, y1_col))
            if j_gp[j] == j_gp[k]:
                gk = j_gkey[j]
                t5_slot[e] = slot_of[gk]
                t5_ids[e] = _prod_col(gk, (cj, y1_col), (ck, y1_col))

        cell_asm[(g, spec["t"])] = {
            "H": H,
            "iu": iu,
            "ju": ju,
            "t1_col": t1_col,
            "has_t34": has_t34,
            "t2_ids": t2_ids,
            "t34_row_ids": t34_row_ids,
            "t34_col_ids": t34_col_ids,
            "gkeys": cell_gkeys,
            "t5_slot": t5_slot,
            "t5_ids": t5_ids,
        }

    registry: Dict[float, Dict] = {}
    for gkey, dmap in diff_index.items():
        pmap = prod_index[gkey]
        diff_keys = np.array(sorted(dmap, key=lambda k: dmap[k]), dtype=np.int64)
        prods = sorted(pmap, key=lambda k: pmap[k])
        registry[gkey] = {
            "diff_keys": diff_keys,
            "prod_i": np.array([p[0] for p in prods], dtype=np.int64),
            "prod_j": np.array([p[1] for p in prods], dtype=np.int64),
            "zero_col": len(prods),
        }

    # Finalize: resolve -1 mask sentinels to each table's appended all-zero
    # column, then convert global column ids to compact (unique columns +
    # per-entry local map) form for cache-resident gathers at assembly time.
    for (g, _t), asm in cell_asm.items():
        asm["t2_cols"], asm["t2_loc"] = np.unique(asm.pop("t2_ids"), return_inverse=True)
        rows = asm.pop("t34_row_ids")
        cols = asm.pop("t34_col_ids")
        if asm.pop("has_t34"):
            zc = registry[g]["zero_col"]
            rows[rows == -1] = zc
            cols[cols == -1] = zc
            both, inv = np.unique(np.concatenate([rows, cols]), return_inverse=True)
            asm["t34_cols"] = both
            asm["t34_row_loc"] = inv[: rows.size]
            asm["t34_col_loc"] = inv[rows.size :]
        else:
            asm["t34_cols"] = None
        t5_slot = asm.pop("t5_slot")
        t5_ids = asm.pop("t5_ids")
        t5_loc = np.empty(t5_slot.size, dtype=np.intp)
        parts = []
        offset = 0
        for s, gk in enumerate(asm.pop("gkeys")):
            sel = t5_slot == s
            u, inv = np.unique(t5_ids[sel], return_inverse=True)
            parts.append((gk, u))
            t5_loc[sel] = offset + inv
            offset += u.size
        t5_loc[t5_slot == -1] = offset
        asm["t5_parts"] = parts
        asm["t5_loc"] = t5_loc
        asm["m5"] = offset

    return registry, cell_asm


def _assemble_omega_tile(
    asm: Dict,
    tables: Dict[float, np.ndarray],
    g: float,
    never_treated_val: float,
    n_rows: int,
) -> np.ndarray:
    """Gather one cell's ``(n_rows, H, H)`` Omega* block from group tables.

    A ``(n_rows, H(H+1)/2)`` upper-triangle strip is accumulated in-place in
    the legacy per-entry operation order (term 1 + term 2, then the term-3/4
    subtractions, then term 5), gathered through compact per-cell column
    slices of the group tables, then written to both triangles (the legacy
    ``omega[k, j] = omega[j, k]`` mirror).  Masked entries hit an all-zero
    column, matching the legacy conditional skips value-exactly; term 1 is
    folded into the compact term-2 slice before the strip gather
    (commutativity-exact, saves a full-width pass).
    """
    table_g = tables[g]
    t2c = tables[never_treated_val][:, asm["t2_cols"]]
    t2c += table_g[:, asm["t1_col"]][:, np.newaxis]
    tri = t2c[:, asm["t2_loc"]]
    if asm["t34_cols"] is not None:
        t34c = table_g[:, asm["t34_cols"]]
        tri -= t34c[:, asm["t34_row_loc"]]
        tri -= t34c[:, asm["t34_col_loc"]]
    m5 = asm["m5"]
    a5 = np.empty((n_rows, m5 + 1))
    off = 0
    for gkey, cols_gk in asm["t5_parts"]:
        a5[:, off : off + cols_gk.size] = tables[gkey][:, cols_gk]
        off += cols_gk.size
    a5[:, m5] = 0.0
    tri += a5[:, asm["t5_loc"]]

    H = asm["H"]
    omega_tile = np.empty((n_rows, H, H))
    omega_tile[:, asm["iu"], asm["ju"]] = tri
    omega_tile[:, asm["ju"], asm["iu"]] = tri
    return omega_tile


def _build_group_kcov_table(
    W: np.ndarray,
    y_group: np.ndarray,
    group_registry: Dict,
    s_tile: np.ndarray,
) -> np.ndarray:
    """One group's kernel-covariance table for a unit tile.

    Same construction as :func:`_kcov_batch` (globally pre-centered
    differenced columns through the GEMM identity
    ``KCov = W @ (A0 * B0) - (W @ A0) * (W @ B0)``), with the per-difference
    ``W @ D0`` GEMM shared across all product columns and the product columns
    chunked to ``_KCOV_PRODUCT_CHUNK`` so the (n_group x chunk) temporary is
    bounded.  The finished table is pre-scaled by the group's own s vector
    (every Omega* term's s factor is the table-group's s) and gets an
    appended all-zero column for masked gather entries.

    Returns shape ``(n_rows, nP + 1)``.
    """
    diff_keys = group_registry["diff_keys"]
    prod_i = group_registry["prod_i"]
    prod_j = group_registry["prod_j"]
    n_p = prod_i.size

    D = y_group[:, diff_keys[:, 0]] - y_group[:, diff_keys[:, 1]]
    D -= D.mean(axis=0)
    WD = W @ D

    table = np.empty((W.shape[0], n_p + 1))
    for lo in range(0, n_p, _KCOV_PRODUCT_CHUNK):
        hi = min(lo + _KCOV_PRODUCT_CHUNK, n_p)
        table[:, lo:hi] = W @ (D[:, prod_i[lo:hi]] * D[:, prod_j[lo:hi]])
    table[:, :n_p] -= WD[:, prod_i] * WD[:, prod_j]
    table[:, :n_p] *= s_tile[:, np.newaxis]
    table[:, n_p] = 0.0
    return table


def compute_conditional_cells_tiled(
    cell_specs: List[Dict],
    outcome_wide: np.ndarray,
    covariate_matrix: np.ndarray,
    cohort_masks: Dict[float, np.ndarray],
    never_treated_mask: np.ndarray,
    period_to_col: Dict[float, int],
    cohort_fractions: Dict[float, float],
    m_hat_cache: Dict[Tuple, np.ndarray],
    r_hat_cache: Dict[Tuple[float, float], np.ndarray],
    s_hat_cache: Dict[float, np.ndarray],
    bandwidth: float,
    omega_ridge: float,
    unit_weights: Optional[np.ndarray] = None,
    never_treated_val: float = np.inf,
    tile_bytes: Optional[int] = None,
) -> Dict[Tuple[float, float], Tuple[float, np.ndarray]]:
    """Fused conditional-path estimation for all (g, t) cells, unit-tiled.

    Replaces the legacy per-cell chain (dense ``compute_omega_star_conditional``
    H^2 loop -> per-unit SVD/pinv weights -> EIF) with, per unit-tile:

    1. ONE kernel-covariance table per comparison group, HOISTED across all
       cells (:func:`_collect_kcov_specs` / :func:`_build_group_kcov_table`):
       every Eq 3.12 term is ``s_group * KCov(Y_u1 - Y_v1, Y_u2 - Y_v2 |
       group)`` keyed only by wide-outcome columns, so the per-cell
       H(H+1)/2-pair tables dedup to one table of distinct product columns
       per group per tile (~26x fewer GEMM columns on a PT-All fit).  The
       group's row-normalized kernel weight matrix
       (``_kernel_weights_matrix``) is built and FREED one group at a time,
       so the tile budget is governed by the largest single group rather
       than the sum of all groups.
    2. Per cell, a gather-assembly of the ``(tile, H, H)`` Omega* block from
       the group tables via precomputed index arrays (upper triangle
       mirrored at the index level, preserving the legacy loop's exact
       per-entry operation sequence).
    3. Ridge-regularized batched weights (:func:`_ridge_solve_weights`;
       requires ``omega_ridge > 0`` - the legacy ``omega_ridge=0`` path never
       reaches this function).
    4. DR generated outcomes on the tile rows (reusing
       :func:`compute_generated_outcomes_cov` on row-sliced views - gen_out
       is row-separable across units) and per-unit weighted scores.

    ``att_gt`` is finalized only after ALL tiles complete (survey-weighted
    mean of the full score vector), and ``EIF = scores - att_gt`` is computed
    once from the finalized value - never per tile.

    Cells with ``H == 1`` skip kernel/omega assembly entirely (weights are
    trivially 1); the results are omega-independent, matching the legacy path.

    Parameters
    ----------
    cell_specs : list of dict
        One per estimable cell: ``{"g", "t", "pairs", "t_col", "y1_col"}``
        (``y1_col`` varies per cohort under ``pt_assumption="post"``).
        ``pairs`` must be non-empty.
    tile_bytes : int, optional
        Memory budget for per-tile state; ``None`` resolves the module
        constant ``_TARGET_OMEGA_TILE_BYTES`` at call time.
    (remaining parameters as in ``compute_omega_star_conditional``)

    Returns
    -------
    dict mapping ``(g, t)`` to ``(att_gt, eif_values)``.
    """
    n_units = outcome_wide.shape[0]
    if not cell_specs:
        return {}

    if n_units > 50_000:
        warnings.warn(
            f"Conditional Omega* estimation with n={n_units} units is "
            f"expensive (the kernel weight matrices are intrinsically "
            f"O(n^2) in memory traffic, computed in bounded tiles).",
            UserWarning,
            stacklevel=2,
        )

    if tile_bytes is None:
        tile_bytes = _TARGET_OMEGA_TILE_BYTES

    h_max = max(len(spec["pairs"]) for spec in cell_specs)

    # Cross-cell kcov key registry + per-cell gather specs (fit-level key
    # bookkeeping; the tables they index are rebuilt per tile). Groups whose
    # keys come only from H == 1 cells get no entry, no table, and no kernel
    # matrix (e.g. a pure pt_assumption="post" fit builds none at all).
    registry, cell_asm = _collect_kcov_specs(cell_specs, period_to_col, never_treated_val)

    def _mask_for(gkey: float) -> np.ndarray:
        return never_treated_mask if np.isinf(gkey) else cohort_masks[gkey]

    # Group-level (non-tiled) data for the groups that need tables.
    y_grp: Dict[float, np.ndarray] = {}
    x_grp: Dict[float, np.ndarray] = {}
    w_grp: Dict[float, Optional[np.ndarray]] = {}
    s_all: Dict[float, np.ndarray] = {}
    for gkey in registry:
        mask = _mask_for(gkey)
        y_grp[gkey] = outcome_wide[mask]
        x_grp[gkey] = covariate_matrix[mask]
        w_grp[gkey] = unit_weights[mask] if unit_weights is not None else None
        s_all[gkey] = s_hat_cache.get(
            gkey,
            np.full(n_units, 1.0 / max(cohort_fractions.get(gkey, 1e-10), 1e-10)),
        )

    # Per-tile-row footprint: the (H, H) omega block plus the triu strip and
    # one gather temporary (together <= 2 H^2), the largest single group's
    # kernel-matrix row counted twice (cdist temp + W - groups are built and
    # freed ONE AT A TIME, so the budget is governed by the largest group,
    # not the sum), the held group tables (product columns + WD columns +
    # zero column per group), the widest cell's compact slices, the
    # (tile x chunk) product-GEMM temporary, and small per-cell vectors.
    # The group-side product temp (n_g x chunk) is tile-independent - carve
    # it out of the budget before dividing.
    n_g_max = max((y.shape[0] for y in y_grp.values()), default=0)
    sum_tables = sum(r["zero_col"] + len(r["diff_keys"]) + 1 for r in registry.values())
    max_compact = max(
        (
            asm["t2_cols"].size
            + (asm["t34_cols"].size if asm["t34_cols"] is not None else 0)
            + asm["m5"]
            + 1
            for asm in cell_asm.values()
        ),
        default=0,
    )
    row_bytes = 8 * (
        2 * h_max * h_max + 2 * n_g_max + sum_tables + max_compact + _KCOV_PRODUCT_CHUNK + 4 * h_max
    )
    budget = tile_bytes - 8 * _KCOV_PRODUCT_CHUNK * n_g_max
    tile_units = int(max(1, min(n_units, budget // max(1, row_bytes))))

    scores: Dict[Tuple[float, float], np.ndarray] = {
        (spec["g"], spec["t"]): np.empty(n_units) for spec in cell_specs
    }

    for lo in range(0, n_units, tile_units):
        hi = min(lo + tile_units, n_units)

        # One kcov table per group for this tile, shared across ALL cells.
        # The kernel weight matrix is an argument temporary: built for one
        # group, released when its table returns.
        tables: Dict[float, np.ndarray] = {}
        for gkey in registry:
            tables[gkey] = _build_group_kcov_table(
                _kernel_weights_matrix(
                    covariate_matrix[lo:hi],
                    x_grp[gkey],
                    bandwidth,
                    group_weights=w_grp[gkey],
                ),
                y_grp[gkey],
                registry[gkey],
                s_all[gkey][lo:hi],
            )

        # Row-sliced views of the nuisance caches / masks for gen_out reuse
        masks_tile = {k: v[lo:hi] for k, v in cohort_masks.items()}
        nt_mask_tile = never_treated_mask[lo:hi]
        m_hat_tile = {k: v[lo:hi] for k, v in m_hat_cache.items()}
        r_hat_tile = {k: v[lo:hi] for k, v in r_hat_cache.items()}
        outcome_tile = outcome_wide[lo:hi]

        for spec in cell_specs:
            g = spec["g"]
            t = spec["t"]
            pairs = spec["pairs"]
            y1_col = spec["y1_col"]
            H = len(pairs)

            gen_out_tile = compute_generated_outcomes_cov(
                target_g=g,
                target_t=t,
                valid_pairs=pairs,
                outcome_wide=outcome_tile,
                cohort_masks=masks_tile,
                never_treated_mask=nt_mask_tile,
                period_to_col=period_to_col,
                period_1_col=y1_col,
                cohort_fractions=cohort_fractions,
                m_hat_cache=m_hat_tile,
                r_hat_cache=r_hat_tile,
                never_treated_val=never_treated_val,
            )

            if H == 1:
                scores[(g, t)][lo:hi] = gen_out_tile[:, 0]
                continue

            omega_tile = _assemble_omega_tile(
                cell_asm[(g, t)], tables, g, never_treated_val, hi - lo
            )
            w_units = _ridge_solve_weights(omega_tile, omega_ridge)
            scores[(g, t)][lo:hi] = np.sum(w_units * gen_out_tile, axis=1)

    out: Dict[Tuple[float, float], Tuple[float, np.ndarray]] = {}
    for spec in cell_specs:
        gt = (spec["g"], spec["t"])
        s = scores[gt]
        if unit_weights is not None:
            att_gt = float(np.average(s, weights=unit_weights))
        else:
            att_gt = float(np.mean(s))
        out[gt] = (att_gt, s - att_gt)
    return out


# ---------------------------------------------------------------------------
# Per-unit efficient weights from conditional Omega*
# ---------------------------------------------------------------------------


def _ridge_solve_weights(omega_stack: np.ndarray, omega_ridge: float) -> np.ndarray:
    """Batched ridge-regularized efficient weights for a stack of Omega*.

    Solves ``(Omega_i + lam * max(trace(Omega_i)/H, 0) * I) x = 1`` per unit
    and normalizes ``w_i = x / (1'x)``. The trace-scaled ridge is
    scale-equivariant and O(H) to compute (no SVD); the ``max(..., 0)`` guards
    a machine-noise-negative trace from producing a negative ridge. Special
    cases match the legacy path: an (approximately) all-zero matrix or a
    near-zero denominator yields uniform weights ``1/H``.

    Parameters
    ----------
    omega_stack : ndarray, shape (n, H, H)
        Stack of covariance matrices (H >= 2).
    omega_ridge : float
        Relative ridge scale ``lam > 0``.

    Returns
    -------
    weights : ndarray, shape (n, H)
    """
    n, H, _ = omega_stack.shape
    ones = np.ones(H)
    weights = np.full((n, H), 1.0 / H)

    # np.allclose(omega_i, 0.0) == elementwise |x| <= atol (1e-8), rtol inert
    zero_mask = np.all(np.abs(omega_stack) <= 1e-8, axis=(1, 2))
    rest = np.flatnonzero(~zero_mask)
    if rest.size == 0:
        return weights

    om = omega_stack[rest]
    trace = np.trace(om, axis1=1, axis2=2)
    ridge = omega_ridge * np.maximum(trace / H, 0.0)
    if (
        HAS_RUST_BACKEND
        and _rust_batched_ridge_chol_solve is not None
        and omega_stack.dtype == np.float64
    ):
        # Batched Cholesky in Rust (ridge added in-kernel - no om_ridged
        # temp). Dispatch is on availability + dtype ONLY: any batch-size
        # cutoff would make forced-tiny-tile fits cross-algorithm vs the
        # single-tile fit and break the tile-invariance twins. A non-finite
        # row signals not-SPD (kernel LU fallback / NaN-poisoned exact
        # singularity) - recompute exactly those rows via the legacy chain
        # so their semantics (incl. the pinv arm) match the numpy path.
        num = _rust_batched_ridge_chol_solve(om, ridge)
        bad = np.flatnonzero(~np.isfinite(num).all(axis=1))
        if bad.size:
            num[bad] = _ridge_solve_numpy(om[bad], ridge[bad])
    else:
        num = _ridge_solve_numpy(om, ridge)
    den = num @ ones
    ok = np.abs(den) >= 1e-15
    solved = np.full((rest.size, H), 1.0 / H)
    solved[ok] = num[ok] / den[ok, None]
    weights[rest] = solved
    return weights


def _ridge_solve_numpy(om: np.ndarray, ridge: np.ndarray) -> np.ndarray:
    """Legacy numpy solve chain for ``(om_i + ridge_i * I) x = 1``.

    Batched LU solve; on an exactly-singular batch, per-unit solve with a
    minimum-norm pseudoinverse backstop (unreachable for PSD Omega* with
    lam > 0, kept so one pathological unit cannot fail the batch). This is
    both the pure-Python path and the per-row recompute target for rows the
    Rust kernel flags as non-finite.
    """
    m_rows, H, _ = om.shape
    om_ridged = om + ridge[:, None, None] * np.eye(H)[None]
    try:
        # gufunc solve treats a 2-D rhs as a matrix; a stack of vectors needs
        # shape (m, H, 1)
        return np.linalg.solve(om_ridged, np.ones((m_rows, H, 1)))[..., 0]
    except np.linalg.LinAlgError:
        ones = np.ones(H)
        num = np.empty((m_rows, H))
        for m in range(m_rows):
            try:
                num[m] = np.linalg.solve(om_ridged[m], ones)
            except np.linalg.LinAlgError:
                num[m] = np.linalg.pinv(om_ridged[m]) @ ones
        return num


def compute_per_unit_weights(
    omega_conditional: np.ndarray,
    cond_threshold: float = 1e12,
    omega_ridge: float = 0.0,
) -> np.ndarray:
    """Per-unit efficient weights from conditional Omega* inverse.

    ``w(X_i) = 1' Omega*(X_i)^{-1} / (1' Omega*(X_i)^{-1} 1)``

    With ``omega_ridge = 0`` (this helper's default), runs the legacy per-unit
    loop: exact inverse, falling back to a pseudoinverse when the condition
    number exceeds ``cond_threshold``. On the numerically singular Omega*
    produced by PT-All's telescoping moments, that pseudoinverse is
    cutoff-cliff sensitive (see ``OMEGA_RIDGE_DEFAULT``). With
    ``omega_ridge > 0``, uses the batched ridge solve
    ``(Omega_i + lam * max(trace/H, 0) * I) x = 1`` instead - numerically
    stable and vectorized over units. The estimator passes its
    ``omega_ridge`` parameter (default ``OMEGA_RIDGE_DEFAULT``); standalone
    callers of this helper keep exact legacy behavior unless they opt in.

    Parameters
    ----------
    omega_conditional : ndarray, shape (n_units, H, H)
        Per-unit conditional covariance matrices.
    cond_threshold : float
        Condition number threshold for pseudoinverse fallback (legacy path).
    omega_ridge : float
        Relative ridge scale; 0 = legacy inv/pinv path.

    Returns
    -------
    weights : ndarray, shape (n_units, H)
        Per-unit efficient combination weights (each row sums to 1).
    """
    n_units, H, _ = omega_conditional.shape
    if H == 0:
        return np.empty((n_units, 0))
    if H == 1:
        return np.ones((n_units, 1))

    if omega_ridge > 0:
        return _ridge_solve_weights(omega_conditional, omega_ridge)

    ones = np.ones(H)
    weights = np.zeros((n_units, H))

    for i in range(n_units):
        omega_i = omega_conditional[i]

        if np.allclose(omega_i, 0.0):
            weights[i] = ones / H
            continue

        cond = float(np.linalg.cond(omega_i))
        if cond > cond_threshold:
            omega_inv = np.linalg.pinv(omega_i)
        else:
            try:
                omega_inv = np.linalg.inv(omega_i)
            except np.linalg.LinAlgError:
                omega_inv = np.linalg.pinv(omega_i)

        numerator = ones @ omega_inv
        denominator = numerator @ ones

        if abs(denominator) < 1e-15:
            weights[i] = ones / H
        else:
            weights[i] = numerator / denominator

    return weights


# ---------------------------------------------------------------------------
# EIF computation
# ---------------------------------------------------------------------------


def compute_eif_cov(
    weights: np.ndarray,
    generated_outcomes: np.ndarray,
    att_gt: float,
    n_units: int,
) -> np.ndarray:
    """Per-unit efficient influence function from DR generated outcomes.

    Supports both global weights ``(H,)`` and per-unit weights ``(n_units, H)``.

    For global weights: ``EIF_i = w @ (gen_out_i - y_bar) = w @ gen_out_i - ATT``
    For per-unit weights: ``EIF_i = w(X_i) @ gen_out_i - ATT``

    In both cases the EIF centers on the scalar ATT estimate, ensuring
    ``mean(EIF) ≈ 0``. The plug-in EIF treats estimated per-unit weights
    as fixed, valid under Neyman orthogonality (Remark 4.2).

    Parameters
    ----------
    weights : ndarray, shape (H,) or (n_units, H)
        Efficient combination weights.
    generated_outcomes : ndarray, shape (n_units, H)
        Per-unit generated outcomes.
    att_gt : float
        Scalar ATT estimate for this (g, t) cell.
    n_units : int
        Total number of units.

    Returns
    -------
    eif : ndarray, shape (n_units,)
        EIF value for every unit. Sample mean is approximately zero.
    """
    if weights.size == 0:
        return np.zeros(n_units)

    if weights.ndim == 1:
        # Global weights: w @ gen_out_i for each unit
        weighted_scores = generated_outcomes @ weights  # (n_units,)
    else:
        # Per-unit weights: w_i @ gen_out_i for each unit
        weighted_scores = np.sum(weights * generated_outcomes, axis=1)

    # Center on the scalar ATT estimate (ensures mean(EIF) ≈ 0)
    eif = weighted_scores - att_gt
    return eif
