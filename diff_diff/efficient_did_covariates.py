"""
Doubly robust math for the Efficient DiD estimator (with covariates).

Implements the with-covariates path from Chen, Sant'Anna & Xie (2025):
OLS outcome regression (linear working model), sieve-based propensity
score ratios (Eq 4.1-4.2), sieve-based inverse propensities (step 4),
kernel-smoothed conditional Omega*(X) for per-unit efficient weights,
doubly robust generated outcomes (Eq 4.4), and the efficient influence
function for analytical standard errors.

The DR property ensures consistency if either the OLS outcome model or
the sieve propensity ratio is correctly specified.  The OLS working model
does not generically guarantee the semiparametric efficiency bound unless
the conditional mean is linear in covariates (see REGISTRY.md).

All functions are pure (no state), operating on pre-pivoted numpy arrays.
"""

import warnings
from itertools import combinations_with_replacement
from math import comb
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist

from diff_diff.linalg import solve_ols

# ---------------------------------------------------------------------------
# Outcome regression
# ---------------------------------------------------------------------------


def estimate_outcome_regression(
    outcome_wide: np.ndarray,
    covariate_matrix: np.ndarray,
    group_mask: np.ndarray,
    t_col: int,
    tpre_col: int,
) -> np.ndarray:
    """Estimate conditional mean outcome change m_hat(X) for a comparison group.

    Regresses ``(Y_t - Y_{tpre})`` on ``X`` within the units identified by
    ``group_mask`` using OLS.  Returns predicted values ``m_hat(X_i)`` for
    **all** units (extrapolated from the within-group fit).

    This implements ``m_hat_{g',t,tpre}(X) = E[Y_t - Y_{tpre} | G=g', X]``.

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

    Returns
    -------
    m_hat : ndarray, shape (n_units,)
        Predicted ``E[Y_t - Y_{tpre} | X]`` for every unit.
    """
    Y_group = outcome_wide[group_mask]
    delta_y = Y_group[:, t_col] - Y_group[:, tpre_col]

    X_group = covariate_matrix[group_mask]
    X_design = np.column_stack([np.ones(len(X_group)), X_group])

    coef, _, _ = solve_ols(
        X_design,
        delta_y,
        return_vcov=False,
        rank_deficient_action="warn",
    )

    X_all = np.column_stack([np.ones(len(covariate_matrix)), covariate_matrix])
    coef_safe = np.where(np.isfinite(coef), coef, 0.0)
    m_hat = X_all @ coef_safe

    non_finite = ~np.isfinite(m_hat)
    if non_finite.any():
        n_bad = int(non_finite.sum())
        warnings.warn(
            f"Outcome regression produced {n_bad} non-finite prediction(s). "
            "Setting to 0.0 (equivalent to no covariate adjustment).",
            UserWarning,
            stacklevel=2,
        )
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

    # Standardize for numerical stability
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


def estimate_propensity_ratio_sieve(
    covariate_matrix: np.ndarray,
    mask_g: np.ndarray,
    mask_gp: np.ndarray,
    k_max: Optional[int] = None,
    criterion: str = "bic",
    ratio_clip: float = 20.0,
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

    On singular basis: tries lower K.  Short-circuits r_{g,g}(X) = 1.

    Parameters
    ----------
    covariate_matrix : ndarray, shape (n_units, n_covariates)
    mask_g : ndarray of bool, shape (n_units,)
        Target treatment group mask.
    mask_gp : ndarray of bool, shape (n_units,)
        Comparison group mask.
    k_max : int or None
        Maximum polynomial degree. None = ``min(floor(n_gp^{1/5}), 5)``.
    criterion : str
        ``"aic"`` or ``"bic"``.
    ratio_clip : float
        Clip ratios to ``[1/ratio_clip, ratio_clip]``.

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

    # Default k_max: use comparison group size, not total n
    if k_max is None:
        k_max = min(int(n_gp**0.2), 5)
    k_max = max(k_max, 1)

    # Penalty multiplier for IC
    n_total = int(np.sum(mask_g)) + n_gp
    c_n = 2.0 if criterion == "aic" else np.log(max(n_total, 2))

    best_ic = np.inf
    best_ratio = np.ones(n_units)  # fallback: constant ratio 1

    for K in range(1, k_max + 1):
        n_basis = comb(K + d, d)

        # Cap K so basis dimension < n_gp (avoid singular system)
        if n_basis >= n_gp:
            break

        basis_all = _polynomial_sieve_basis(covariate_matrix, K)
        Psi_gp = basis_all[mask_gp]  # (n_gp, n_basis)
        Psi_g = basis_all[mask_g]  # (n_g, n_basis)

        # Normal equations: (Psi_gp' Psi_gp) beta = Psi_g.sum(axis=0)
        A = Psi_gp.T @ Psi_gp
        b = Psi_g.sum(axis=0)

        try:
            beta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            continue  # singular — try next K

        # Check for NaN/Inf in solution
        if not np.all(np.isfinite(beta)):
            continue

        # Predicted ratio for all units
        r_hat = basis_all @ beta

        # IC selection: loss at optimum = -(1/n) * b'beta
        # Derivation: L(beta) = (1/n)(beta'A*beta - 2*b'beta).
        # At optimum A*beta = b, so beta'A*beta = b'beta.
        # Therefore L = (1/n)(b'beta - 2*b'beta) = -(1/n)*b'beta.
        loss = -float(b @ beta) / n_total
        ic_val = 2.0 * loss + c_n * n_basis / n_total

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

    Parameters
    ----------
    covariate_matrix : ndarray, shape (n_units, n_covariates)
    group_mask : ndarray of bool, shape (n_units,)
        Mask for the group whose inverse propensity to estimate.
    k_max : int or None
        Maximum polynomial degree. None = auto.
    criterion : str
        ``"aic"`` or ``"bic"``.

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

    if k_max is None:
        k_max = min(int(n_group**0.2), 5)
    k_max = max(k_max, 1)

    c_n = 2.0 if criterion == "aic" else np.log(max(n_units, 2))

    best_ic = np.inf
    best_s = np.full(n_units, n_units / n_group)  # fallback: unconditional

    for K in range(1, k_max + 1):
        n_basis = comb(K + d, d)
        if n_basis >= n_group:
            break

        basis_all = _polynomial_sieve_basis(covariate_matrix, K)
        Psi_gp = basis_all[group_mask]

        A = Psi_gp.T @ Psi_gp
        # RHS: sum of basis over ALL units (not just one group)
        b = basis_all.sum(axis=0)

        try:
            beta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            continue
        if not np.all(np.isfinite(beta)):
            continue

        s_hat = basis_all @ beta

        # IC: loss = -(1/n) * b'beta (same derivation as ratio estimator)
        loss = -float(b @ beta) / n_units
        ic_val = 2.0 * loss + c_n * n_basis / n_units

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


def _silverman_bandwidth(X: np.ndarray) -> float:
    """Silverman's rule-of-thumb bandwidth for d-dimensional X.

    ``h = (4 / (d + 2))^{1/(d+4)} * median_std * n^{-1/(d+4)}``
    """
    n, d = X.shape
    stds = np.std(X, axis=0)
    stds[stds < 1e-10] = 1.0
    median_std = float(np.median(stds))
    h = (4.0 / (d + 2)) ** (1.0 / (d + 4)) * median_std * n ** (-1.0 / (d + 4))
    return max(h, 1e-10)


def _kernel_weights_matrix(
    X_all: np.ndarray,
    X_group: np.ndarray,
    bandwidth: float,
) -> np.ndarray:
    """Gaussian kernel weight matrix.

    Returns shape ``(n_all, n_group)`` where entry ``[i, j]`` is the
    normalized kernel weight ``K_h(X_group[j], X_all[i])``.

    Each row sums to 1 (Nadaraya-Watson normalization).
    """
    # Squared distances: (n_all, n_group)
    dist_sq = cdist(X_all, X_group, metric="sqeuclidean")
    # Gaussian kernel
    raw = np.exp(-dist_sq / (2.0 * bandwidth**2))
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
        bandwidth = _silverman_bandwidth(covariate_matrix)

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

    # Pre-compute kernel weight matrices per group
    Y_g = outcome_wide[g_mask]
    X_g = covariate_matrix[g_mask]
    Yg_t_minus_1 = Y_g[:, t_col] - Y_g[:, y1_col]

    W_g = _kernel_weights_matrix(covariate_matrix, X_g, bandwidth)
    W_inf = _kernel_weights_matrix(covariate_matrix, X_inf, bandwidth)

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
                        W_gp_cache[gp_j] = _kernel_weights_matrix(covariate_matrix, X_gp, bandwidth)
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
# Per-unit efficient weights from conditional Omega*
# ---------------------------------------------------------------------------


def compute_per_unit_weights(
    omega_conditional: np.ndarray,
    cond_threshold: float = 1e12,
) -> np.ndarray:
    """Per-unit efficient weights from conditional Omega* inverse.

    ``w(X_i) = 1' Omega*(X_i)^{-1} / (1' Omega*(X_i)^{-1} 1)``

    Falls back to pseudoinverse per unit if condition number exceeds threshold.

    Parameters
    ----------
    omega_conditional : ndarray, shape (n_units, H, H)
        Per-unit conditional covariance matrices.
    cond_threshold : float
        Condition number threshold for pseudoinverse fallback.

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
