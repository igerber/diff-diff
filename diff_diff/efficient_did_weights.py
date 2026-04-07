"""
Mathematical core for the Efficient DiD estimator.

Implements the no-covariates path from Chen, Sant'Anna & Xie (2025):
optimal weighting via the inverse of the conditional covariance matrix Omega*,
generated outcomes from within-group sample means, and the efficient
influence function for analytical standard errors.

All functions are pure (no state), operating on pre-pivoted numpy arrays.
"""

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np


def enumerate_valid_triples(
    target_g: float,
    treatment_groups: List[float],
    time_periods: List[float],
    period_1: float,
    pt_assumption: str,
    anticipation: int = 0,
    never_treated_val: float = np.inf,
) -> List[Tuple[float, float]]:
    """Enumerate valid (g', t_pre) pairs for target (g, t).

    Under PT-All, any not-yet-treated cohort g' (including never-treated and
    g'=g itself) paired with any baseline t_pre that is pre-treatment for the
    *comparison* group g' forms a valid comparison.  The target group g appears
    only in the first term (Y_t - Y_1), which is independent of t_pre, so
    t_pre need not be pre-treatment for g.  Under PT-Post, only the
    never-treated group with baseline g - 1 - anticipation is valid
    (just-identified).

    Parameters
    ----------
    target_g : float
        Treatment cohort of the target group.
    treatment_groups : list of float
        All treatment cohort identifiers (finite values only).
    time_periods : list of float
        All observed time periods, sorted.
    period_1 : float
        Earliest observed period (universal baseline).
    pt_assumption : str
        ``"all"`` or ``"post"``.
    anticipation : int
        Number of anticipation periods.
    never_treated_val : float
        Sentinel for the never-treated group (default ``np.inf``).

    Returns
    -------
    list of (g', t_pre) tuples
        Valid comparison pairs.  Empty if none exist.
    """
    if pt_assumption == "post":
        # Just-identified: only (never-treated, g - 1 - anticipation)
        baseline = target_g - 1 - anticipation
        if baseline >= period_1:
            return [(never_treated_val, baseline)]
        return []

    # PT-All: overidentified
    pairs: List[Tuple[float, float]] = []

    # Candidate comparison groups: never-treated + all treatment cohorts.
    # Including g'=g (same-cohort) is valid under PT-All (Eq 3.9).
    # Including g'=∞ (never-treated) produces moments where the second
    # and third terms telescope: y_hat = E[Y_t-Y_1|G=g] - E[Y_t-Y_1|G=∞]
    # regardless of t_pre. These redundant moments add no information
    # beyond the basic 2x2 DiD; Omega*'s pseudoinverse assigns them
    # zero effective weight. Retained for implementation simplicity.
    candidate_groups: List[float] = [never_treated_val]
    for gp in treatment_groups:
        candidate_groups.append(gp)

    for gp in candidate_groups:
        # Determine effective treatment start for comparison group
        if np.isinf(gp):
            effective_gp = np.inf  # never treated
        else:
            effective_gp = gp - anticipation

        for t_pre in time_periods:
            if t_pre == period_1:
                # period_1 is the universal reference — used as Y_1 in the
                # differencing (Eq 3.9 first term). Including t_pre = period_1
                # would make the third term Y_1 - Y_1 = 0 (degenerate), so it
                # adds no information to Omega* regardless of which g' is used.
                continue
            # Only require t_pre < g' (pre-treatment for comparison group).
            # No constraint on t_pre vs g: the target group appears only in
            # the first term (Y_t - Y_1), which is independent of t_pre.
            if not np.isinf(effective_gp) and t_pre >= effective_gp:
                continue
            pairs.append((gp, t_pre))

    return pairs


def _sample_cov(
    a: np.ndarray,
    b: np.ndarray,
    w: Optional[np.ndarray] = None,
) -> float:
    """Sample covariance between two 1-D arrays (ddof=1).

    Returns 0.0 if fewer than 2 observations.

    Parameters
    ----------
    a, b : ndarray, shape (n,)
        Data arrays.
    w : ndarray, shape (n,), optional
        Survey weights.  When provided, computes the reliability-weighted
        covariance: ``sum(w*(a-a_bar)*(b-b_bar)) / (sum(w) - 1)`` where
        ``a_bar = average(a, weights=w)``.
    """
    n = len(a)
    if n < 2:
        return 0.0
    if w is None:
        return float(((a - a.mean()) * (b - b.mean())).sum() / (n - 1))
    # Weighted covariance with reliability weights (Bessel-style correction)
    a_bar = float(np.average(a, weights=w))
    b_bar = float(np.average(b, weights=w))
    sum_w = float(np.sum(w))
    if sum_w <= 1.0:
        return 0.0
    return float(np.sum(w * (a - a_bar) * (b - b_bar)) / (sum_w - 1.0))


def compute_omega_star_nocov(
    target_g: float,
    target_t: float,
    valid_pairs: List[Tuple[float, float]],
    outcome_wide: np.ndarray,
    cohort_masks: Dict[float, np.ndarray],
    never_treated_mask: np.ndarray,
    period_to_col: Dict[float, int],
    period_1_col: int,
    cohort_fractions: Dict[float, float],
    never_treated_val: float = np.inf,
    unit_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build the |H| x |H| covariance matrix Omega* (Eq 3.12, unconditional).

    Each element Omega*[j,k] is the sum of up to five covariance terms
    computed from within-group sample covariances scaled by inverse
    cohort fractions.

    Parameters
    ----------
    target_g : float
        Target treatment cohort.
    target_t : float
        Target time period.
    valid_pairs : list of (g', t_pre) tuples
        Valid comparison pairs from :func:`enumerate_valid_triples`.
    outcome_wide : ndarray, shape (n_units, n_periods)
        Pivoted outcome matrix.
    cohort_masks : dict
        ``{cohort: bool_mask}`` over the unit dimension.
    never_treated_mask : ndarray of bool
        Mask for never-treated units.
    period_to_col : dict
        ``{period: column_index}`` in ``outcome_wide``.
    period_1_col : int
        Column index of the earliest period (universal baseline Y_1).
    cohort_fractions : dict
        ``{cohort: n_cohort / n}`` for each cohort.
    never_treated_val : float
        Sentinel for the never-treated group.
    unit_weights : ndarray, shape (n_units,), optional
        Survey weights at the unit level.  When provided, all sample
        means and covariances are weighted.

    Returns
    -------
    ndarray, shape (|H|, |H|)
        Covariance matrix.  Empty (0,0) array if ``valid_pairs`` is empty.
    """
    H = len(valid_pairs)
    if H == 0:
        return np.empty((0, 0))

    t_col = period_to_col[target_t]
    y1_col = period_1_col

    # Pre-extract outcome columns for target group g
    g_mask = cohort_masks[target_g]
    Y_g = outcome_wide[g_mask]  # (n_g, n_periods)
    pi_g = cohort_fractions[target_g]

    # Extract per-cohort weights (None propagates = unweighted)
    w_g = unit_weights[g_mask] if unit_weights is not None else None
    w_inf = unit_weights[never_treated_mask] if unit_weights is not None else None

    # Y_t - Y_1 for the target group
    Yg_t_minus_1 = Y_g[:, t_col] - Y_g[:, y1_col]

    # Never-treated outcomes
    Y_inf = outcome_wide[never_treated_mask]
    pi_inf = cohort_fractions.get(never_treated_val, 0.0)

    omega = np.zeros((H, H))

    # Hoist Term 1: (1/pi_g) * Var(Y_t - Y_1 | G=g) — same for all (j, k)
    term1 = 0.0
    if pi_g > 0:
        term1 = (1.0 / pi_g) * _sample_cov(Yg_t_minus_1, Yg_t_minus_1, w=w_g)

    # Precompute differenced arrays to avoid redundant slicing in the loop
    # Never-treated: Y_t - Y_{tpre} and Y_{tpre} - Y_1 for each tpre
    inf_t_minus_tpre: Dict[int, np.ndarray] = {}
    inf_tpre_minus_1: Dict[int, np.ndarray] = {}
    if len(Y_inf) >= 2:
        for _, tpre in valid_pairs:
            tpre_col = period_to_col[tpre]
            if tpre_col not in inf_t_minus_tpre:
                inf_t_minus_tpre[tpre_col] = Y_inf[:, t_col] - Y_inf[:, tpre_col]
                inf_tpre_minus_1[tpre_col] = Y_inf[:, tpre_col] - Y_inf[:, y1_col]

    # Target group: Y_{tpre} - Y_1 for each tpre where g' == target_g
    g_tpre_minus_1: Dict[int, np.ndarray] = {}
    if pi_g > 0:
        for gp, tpre in valid_pairs:
            if gp == target_g:
                tpre_col = period_to_col[tpre]
                if tpre_col not in g_tpre_minus_1:
                    g_tpre_minus_1[tpre_col] = Y_g[:, tpre_col] - Y_g[:, y1_col]

    # Comparison cohort submatrices: cache outcome_wide[cohort_masks[gp]]
    gp_outcomes: Dict[float, np.ndarray] = {}
    gp_weights: Dict[float, Optional[np.ndarray]] = {}
    for gp, _ in valid_pairs:
        if not np.isinf(gp) and gp not in gp_outcomes:
            if gp in cohort_masks:
                gp_outcomes[gp] = outcome_wide[cohort_masks[gp]]
                gp_weights[gp] = (
                    unit_weights[cohort_masks[gp]] if unit_weights is not None else None
                )

    # Comparison cohort: Y_{tpre} - Y_1 for each (gp, tpre) pair in Term 5
    gp_tpre_minus_1: Dict[Tuple[float, int], np.ndarray] = {}

    for j in range(H):
        gp_j, tpre_j = valid_pairs[j]
        tpre_j_col = period_to_col[tpre_j]

        for k in range(j, H):
            gp_k, tpre_k = valid_pairs[k]
            tpre_k_col = period_to_col[tpre_k]

            val = term1

            # Term 2: (1/pi_inf) * SampleCov(Y_t - Y_{tpre_j}, Y_t - Y_{tpre_k} | G=inf)
            if pi_inf > 0 and tpre_j_col in inf_t_minus_tpre:
                val += (1.0 / pi_inf) * _sample_cov(
                    inf_t_minus_tpre[tpre_j_col],
                    inf_t_minus_tpre[tpre_k_col],
                    w=w_inf,
                )

            # Term 3: -1{g == g'_j} / pi_g * SampleCov(Y_t-Y_1, Y_{tpre_j}-Y_1 | G=g)
            if gp_j == target_g and tpre_j_col in g_tpre_minus_1:
                val -= (1.0 / pi_g) * _sample_cov(
                    Yg_t_minus_1,
                    g_tpre_minus_1[tpre_j_col],
                    w=w_g,
                )

            # Term 4: -1{g == g'_k} / pi_g * SampleCov(Y_t-Y_1, Y_{tpre_k}-Y_1 | G=g)
            if gp_k == target_g and tpre_k_col in g_tpre_minus_1:
                val -= (1.0 / pi_g) * _sample_cov(
                    Yg_t_minus_1,
                    g_tpre_minus_1[tpre_k_col],
                    w=w_g,
                )

            # Term 5: 1{g'_j == g'_k} / pi_{g'_j} * SampleCov(Y_{tpre_j}-Y_1, Y_{tpre_k}-Y_1 | G=g'_j)
            if gp_j == gp_k:
                if np.isinf(gp_j):
                    if pi_inf > 0 and tpre_j_col in inf_tpre_minus_1:
                        val += (1.0 / pi_inf) * _sample_cov(
                            inf_tpre_minus_1[tpre_j_col],
                            inf_tpre_minus_1[tpre_k_col],
                            w=w_inf,
                        )
                else:
                    pi_gp = cohort_fractions.get(gp_j, 0.0)
                    if pi_gp > 0 and gp_j in cohort_masks:
                        Y_gp = gp_outcomes.get(gp_j)
                        if Y_gp is None:
                            Y_gp = outcome_wide[cohort_masks[gp_j]]
                        w_gp = gp_weights.get(gp_j)
                        if len(Y_gp) >= 2:
                            # Cache tpre diffs for comparison cohorts
                            key_j = (gp_j, tpre_j_col)
                            if key_j not in gp_tpre_minus_1:
                                gp_tpre_minus_1[key_j] = Y_gp[:, tpre_j_col] - Y_gp[:, y1_col]
                            key_k = (gp_j, tpre_k_col)
                            if key_k not in gp_tpre_minus_1:
                                gp_tpre_minus_1[key_k] = Y_gp[:, tpre_k_col] - Y_gp[:, y1_col]
                            val += (1.0 / pi_gp) * _sample_cov(
                                gp_tpre_minus_1[key_j],
                                gp_tpre_minus_1[key_k],
                                w=w_gp,
                            )

            omega[j, k] = val
            if j != k:
                omega[k, j] = val

    return omega


def compute_efficient_weights(
    omega_star: np.ndarray,
    cond_threshold: float = 1e12,
) -> Tuple[np.ndarray, bool, float]:
    """Compute efficient weights from Omega* inverse (Eq 3.13 / 4.3).

    ``w = ones @ inv(Omega*) / (ones @ inv(Omega*) @ ones)``

    Parameters
    ----------
    omega_star : ndarray, shape (H, H)
        Covariance matrix from :func:`compute_omega_star_nocov`.
    cond_threshold : float
        If condition number exceeds this, use pseudoinverse + warning.

    Returns
    -------
    weights : ndarray, shape (H,)
        Efficient combination weights (sum to 1).
    used_pinv : bool
        True if pseudoinverse was used.
    cond_number : float
        Condition number of Omega* (avoids recomputation by caller).
    """
    H = omega_star.shape[0]
    if H == 0:
        return np.array([]), False, 0.0
    if H == 1:
        return np.array([1.0]), False, 1.0

    ones = np.ones(H)
    used_pinv = False

    # Check for zero matrix
    if np.allclose(omega_star, 0.0):
        warnings.warn(
            "Omega* matrix is all zeros; using uniform weights.",
            UserWarning,
            stacklevel=2,
        )
        return ones / H, False, np.inf

    cond = float(np.linalg.cond(omega_star))
    if cond > cond_threshold:
        warnings.warn(
            f"Omega* condition number ({cond:.2e}) exceeds threshold "
            f"({cond_threshold:.2e}); using pseudoinverse for weights.",
            UserWarning,
            stacklevel=2,
        )
        omega_inv = np.linalg.pinv(omega_star)
        used_pinv = True
    else:
        try:
            omega_inv = np.linalg.inv(omega_star)
        except np.linalg.LinAlgError:
            omega_inv = np.linalg.pinv(omega_star)
            used_pinv = True

    numerator = ones @ omega_inv  # shape (H,)
    denominator = numerator @ ones  # scalar

    if abs(denominator) < 1e-15:
        warnings.warn(
            "Denominator of efficient weights is near zero; using uniform weights.",
            UserWarning,
            stacklevel=2,
        )
        return ones / H, used_pinv, cond

    weights = numerator / denominator
    return weights, used_pinv, cond


def compute_generated_outcomes_nocov(
    target_g: float,
    target_t: float,
    valid_pairs: List[Tuple[float, float]],
    outcome_wide: np.ndarray,
    cohort_masks: Dict[float, np.ndarray],
    never_treated_mask: np.ndarray,
    period_to_col: Dict[float, int],
    period_1_col: int,
    never_treated_val: float = np.inf,
    unit_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute generated outcome vector (one scalar per valid pair).

    In the no-covariates case each generated outcome is a triple-difference
    of within-group sample means (Eq 3.9 / 4.4 simplified)::

        Y_hat_j = mean(Y_t - Y_1 | G=g)
                - mean(Y_t - Y_{t_pre} | G=inf)
                - mean(Y_{t_pre} - Y_1 | G=g')

    where ``inf`` denotes the never-treated group and ``g'`` is the comparison
    cohort for pair *j*.

    Parameters
    ----------
    target_g, target_t : float
        Target group-time.
    valid_pairs : list of (g', t_pre)
        Valid comparison pairs.
    outcome_wide : ndarray, shape (n_units, n_periods)
    cohort_masks, never_treated_mask, period_to_col, period_1_col :
        Pre-computed data structures.
    never_treated_val : float
        Sentinel for never-treated.
    unit_weights : ndarray, shape (n_units,), optional
        Survey weights at the unit level.  When provided, all sample
        means become weighted means.

    Returns
    -------
    ndarray, shape (|H|,)
        Scalar generated outcome for each pair.
    """
    H = len(valid_pairs)
    if H == 0:
        return np.array([])

    t_col = period_to_col[target_t]
    y1_col = period_1_col

    # Helper: weighted or unweighted mean
    def _wmean(vals: np.ndarray, w: Optional[np.ndarray]) -> float:
        if w is not None:
            return float(np.average(vals, weights=w))
        return float(np.mean(vals))

    # Per-cohort weights
    g_mask = cohort_masks[target_g]
    w_g = unit_weights[g_mask] if unit_weights is not None else None
    w_inf = unit_weights[never_treated_mask] if unit_weights is not None else None

    # Target group mean: mean(Y_t - Y_1 | G = g)
    Y_g = outcome_wide[g_mask]
    mean_g_t_1 = _wmean(Y_g[:, t_col] - Y_g[:, y1_col], w_g)

    # Never-treated outcomes
    Y_inf = outcome_wide[never_treated_mask]

    y_hat = np.empty(H)

    for j, (gp, tpre) in enumerate(valid_pairs):
        tpre_col = period_to_col[tpre]

        # mean(Y_t - Y_{tpre} | G = inf)
        mean_inf_t_tpre = _wmean(Y_inf[:, t_col] - Y_inf[:, tpre_col], w_inf)

        # mean(Y_{tpre} - Y_1 | G = g')
        if np.isinf(gp):
            Y_gp = Y_inf
            w_gp = w_inf
        else:
            Y_gp = outcome_wide[cohort_masks[gp]]
            w_gp = unit_weights[cohort_masks[gp]] if unit_weights is not None else None
        mean_gp_tpre_1 = _wmean(Y_gp[:, tpre_col] - Y_gp[:, y1_col], w_gp)

        y_hat[j] = mean_g_t_1 - mean_inf_t_tpre - mean_gp_tpre_1

    return y_hat


def compute_eif_nocov(
    target_g: float,
    target_t: float,
    weights: np.ndarray,
    valid_pairs: List[Tuple[float, float]],
    outcome_wide: np.ndarray,
    cohort_masks: Dict[float, np.ndarray],
    never_treated_mask: np.ndarray,
    period_to_col: Dict[float, int],
    period_1_col: int,
    cohort_fractions: Dict[float, float],
    n_units: int,
    never_treated_val: float = np.inf,
    unit_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute per-unit efficient influence function values.

    For each unit *i* and each valid pair *j*, three terms contribute to
    the EIF depending on the unit's cohort membership:

    * **Treated term** (unit in cohort g):
      ``(1/pi_g) * (Y_{i,t} - Y_{i,1} - Y_hat_j) - ATT(g,t)``
    * **Never-treated term** (unit in never-treated):
      ``-(1/pi_g) * (1/pi_inf) * pi_g * (Y_{i,t} - Y_{i,tpre_j} - mean_inf)``
      (simplified: contributes the comparison group score for the never-treated)
    * **Comparison cohort term** (unit in cohort g'_j):
      ``-(1/pi_g) * (1/pi_{g'_j}) * pi_g * (Y_{i,tpre_j} - Y_{i,1} - mean_gp)``

    These are combined with efficient weights ``w_j``.

    The derivation follows Theorem 3.2 and Eq 3.9-3.10, simplified for
    the no-covariates case where propensity score ratios equal cohort
    fraction ratios.

    Parameters
    ----------
    target_g, target_t : float
        Target group-time.
    weights : ndarray, shape (H,)
        Efficient weights.
    valid_pairs : list of (g', t_pre)
    outcome_wide, cohort_masks, never_treated_mask, period_to_col,
    period_1_col, cohort_fractions, n_units, never_treated_val :
        Pre-computed data structures.
    unit_weights : ndarray, shape (n_units,), optional
        Survey weights at the unit level.  When provided, within-group
        means are weighted means.

    Returns
    -------
    ndarray, shape (n_units,)
        EIF value for every unit.
    """
    H = len(valid_pairs)
    if H == 0:
        return np.zeros(n_units)

    t_col = period_to_col[target_t]
    y1_col = period_1_col

    g_mask = cohort_masks[target_g]
    Y_g = outcome_wide[g_mask]
    pi_g = cohort_fractions[target_g]

    Y_inf = outcome_wide[never_treated_mask]
    pi_inf = cohort_fractions.get(never_treated_val, 0.0)

    # Per-cohort weights
    w_g = unit_weights[g_mask] if unit_weights is not None else None
    w_inf = unit_weights[never_treated_mask] if unit_weights is not None else None

    # Helper for weighted/unweighted mean
    def _wmean(vals: np.ndarray, w: Optional[np.ndarray]) -> float:
        if w is not None:
            return float(np.average(vals, weights=w))
        return float(np.mean(vals))

    eif = np.zeros(n_units)

    # Hoist treated-group computations out of the per-pair loop (j-invariant)
    Yg_t_minus_1 = Y_g[:, t_col] - Y_g[:, y1_col]
    mean_g_t_1 = _wmean(Yg_t_minus_1, w_g)
    treated_demeaned = None
    if pi_g > 0:
        treated_demeaned = (1.0 / pi_g) * (Yg_t_minus_1 - mean_g_t_1)

    # Precompute never-treated diffs per tpre to avoid recomputation
    inf_diffs: Dict[int, np.ndarray] = {}
    inf_means: Dict[int, float] = {}

    for j, (gp, tpre) in enumerate(valid_pairs):
        w_j = weights[j]
        tpre_col = period_to_col[tpre]

        # --- Treated term (units in cohort g) ---
        # (1/pi_g) * demeaned(Y_t - Y_1 | G=g) — same for all j
        if treated_demeaned is not None:
            eif[g_mask] += w_j * treated_demeaned

        # --- Never-treated term ---
        if tpre_col not in inf_diffs:
            inf_diffs[tpre_col] = Y_inf[:, t_col] - Y_inf[:, tpre_col]
            inf_means[tpre_col] = _wmean(inf_diffs[tpre_col], w_inf)
        if pi_inf > 0:
            inf_contrib = -(1.0 / pi_inf) * (inf_diffs[tpre_col] - inf_means[tpre_col])
            eif[never_treated_mask] += w_j * inf_contrib

        # --- Comparison cohort term ---
        # Contribution from units in cohort g'_j for the baseline shift tpre_j - Y_1
        if np.isinf(gp):
            # Comparison group is never-treated; contribution is folded into
            # the never-treated term via Y_{tpre} - Y_1 differencing.
            # Additional term: -(1/pi_inf) * demeaned (Y_{tpre} - Y_1 | G=inf)
            mean_inf_tpre_1 = _wmean(Y_inf[:, tpre_col] - Y_inf[:, y1_col], w_inf)
            if pi_inf > 0:
                gp_contrib = -(1.0 / pi_inf) * (
                    (Y_inf[:, tpre_col] - Y_inf[:, y1_col]) - mean_inf_tpre_1
                )
                eif[never_treated_mask] += w_j * gp_contrib
        else:
            gp_mask = cohort_masks[gp]
            Y_gp = outcome_wide[gp_mask]
            pi_gp = cohort_fractions.get(gp, 0.0)
            w_gp = unit_weights[gp_mask] if unit_weights is not None else None
            mean_gp_tpre_1 = _wmean(Y_gp[:, tpre_col] - Y_gp[:, y1_col], w_gp)
            if pi_gp > 0:
                gp_contrib = -(1.0 / pi_gp) * (
                    (Y_gp[:, tpre_col] - Y_gp[:, y1_col]) - mean_gp_tpre_1
                )
                eif[gp_mask] += w_j * gp_contrib

    return eif
