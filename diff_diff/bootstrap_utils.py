"""
Shared bootstrap utilities for multiplier bootstrap inference.

Provides weight generation, percentile CI, and p-value helpers used by
both CallawaySantAnna and ContinuousDiD estimators.
"""

import warnings
from typing import Optional, Tuple

import numpy as np

from diff_diff._backend import HAS_RUST_BACKEND, _rust_bootstrap_weights

__all__ = [
    "generate_bootstrap_weights",
    "generate_bootstrap_weights_batch",
    "generate_bootstrap_weights_batch_numpy",
    "generate_survey_multiplier_weights_batch",
    "generate_rao_wu_weights",
    "generate_rao_wu_weights_batch",
    "compute_percentile_ci",
    "compute_bootstrap_pvalue",
    "compute_effect_bootstrap_stats",
    "compute_effect_bootstrap_stats_batch",
]


def generate_bootstrap_weights(
    n_units: int,
    weight_type: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate bootstrap weights for multiplier bootstrap.

    Parameters
    ----------
    n_units : int
        Number of units (clusters) to generate weights for.
    weight_type : str
        Type of weights: "rademacher", "mammen", or "webb".
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Array of bootstrap weights with shape (n_units,).
    """
    if weight_type == "rademacher":
        return rng.choice([-1.0, 1.0], size=n_units)
    elif weight_type == "mammen":
        sqrt5 = np.sqrt(5)
        val1 = -(sqrt5 - 1) / 2
        val2 = (sqrt5 + 1) / 2
        p1 = (sqrt5 + 1) / (2 * sqrt5)
        return rng.choice([val1, val2], size=n_units, p=[p1, 1 - p1])
    elif weight_type == "webb":
        values = np.array(
            [
                -np.sqrt(3 / 2),
                -np.sqrt(2 / 2),
                -np.sqrt(1 / 2),
                np.sqrt(1 / 2),
                np.sqrt(2 / 2),
                np.sqrt(3 / 2),
            ]
        )
        return rng.choice(values, size=n_units)
    else:
        raise ValueError(
            f"weight_type must be 'rademacher', 'mammen', or 'webb', " f"got '{weight_type}'"
        )


def generate_bootstrap_weights_batch(
    n_bootstrap: int,
    n_units: int,
    weight_type: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate all bootstrap weights at once (vectorized).

    Uses Rust backend if available for parallel generation.

    Parameters
    ----------
    n_bootstrap : int
        Number of bootstrap iterations.
    n_units : int
        Number of units (clusters) to generate weights for.
    weight_type : str
        Type of weights: "rademacher", "mammen", or "webb".
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Array of bootstrap weights with shape (n_bootstrap, n_units).
    """
    if HAS_RUST_BACKEND and _rust_bootstrap_weights is not None:
        seed = rng.integers(0, 2**63 - 1)
        return _rust_bootstrap_weights(n_bootstrap, n_units, weight_type, seed)
    return generate_bootstrap_weights_batch_numpy(n_bootstrap, n_units, weight_type, rng)


def generate_bootstrap_weights_batch_numpy(
    n_bootstrap: int,
    n_units: int,
    weight_type: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    NumPy fallback implementation of :func:`generate_bootstrap_weights_batch`.

    Parameters
    ----------
    n_bootstrap : int
        Number of bootstrap iterations.
    n_units : int
        Number of units (clusters) to generate weights for.
    weight_type : str
        Type of weights: "rademacher", "mammen", or "webb".
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Array of bootstrap weights with shape (n_bootstrap, n_units).
    """
    if weight_type == "rademacher":
        return rng.choice([-1.0, 1.0], size=(n_bootstrap, n_units))
    elif weight_type == "mammen":
        sqrt5 = np.sqrt(5)
        val1 = -(sqrt5 - 1) / 2
        val2 = (sqrt5 + 1) / 2
        p1 = (sqrt5 + 1) / (2 * sqrt5)
        return rng.choice([val1, val2], size=(n_bootstrap, n_units), p=[p1, 1 - p1])
    elif weight_type == "webb":
        values = np.array(
            [
                -np.sqrt(3 / 2),
                -np.sqrt(2 / 2),
                -np.sqrt(1 / 2),
                np.sqrt(1 / 2),
                np.sqrt(2 / 2),
                np.sqrt(3 / 2),
            ]
        )
        return rng.choice(values, size=(n_bootstrap, n_units))
    else:
        raise ValueError(
            f"weight_type must be 'rademacher', 'mammen', or 'webb', " f"got '{weight_type}'"
        )


def compute_percentile_ci(
    boot_dist: np.ndarray,
    alpha: float,
) -> Tuple[float, float]:
    """
    Compute percentile confidence interval from bootstrap distribution.

    Parameters
    ----------
    boot_dist : np.ndarray
        Bootstrap distribution (1-D array).
    alpha : float
        Significance level (e.g., 0.05 for 95% CI).

    Returns
    -------
    tuple of float
        ``(lower, upper)`` confidence interval bounds.
    """
    lower = float(np.percentile(boot_dist, alpha / 2 * 100))
    upper = float(np.percentile(boot_dist, (1 - alpha / 2) * 100))
    return (lower, upper)


def compute_bootstrap_pvalue(
    original_effect: float,
    boot_dist: np.ndarray,
    n_valid: Optional[int] = None,
) -> float:
    """
    Compute two-sided bootstrap p-value using the percentile method.

    Parameters
    ----------
    original_effect : float
        Original point estimate.
    boot_dist : np.ndarray
        Bootstrap distribution of the effect.
    n_valid : int, optional
        Number of valid bootstrap samples for p-value floor.
        If None, uses ``len(boot_dist)``.

    Returns
    -------
    float
        Two-sided bootstrap p-value.
    """
    if original_effect >= 0:
        p_one_sided = np.mean(boot_dist <= 0)
    else:
        p_one_sided = np.mean(boot_dist >= 0)

    p_value = min(2 * p_one_sided, 1.0)
    n_for_floor = n_valid if n_valid is not None else len(boot_dist)
    p_value = max(p_value, 1 / (n_for_floor + 1))
    return float(p_value)


def compute_effect_bootstrap_stats(
    original_effect: float,
    boot_dist: np.ndarray,
    alpha: float = 0.05,
    context: str = "bootstrap distribution",
) -> Tuple[float, Tuple[float, float], float]:
    """
    Compute bootstrap statistics for a single effect.

    Filters non-finite samples, returning NaN for all statistics if
    fewer than 50% of samples are valid.

    Parameters
    ----------
    original_effect : float
        Original point estimate.
    boot_dist : np.ndarray
        Bootstrap distribution of the effect.
    alpha : float, default=0.05
        Significance level.
    context : str, optional
        Description for warning messages.

    Returns
    -------
    se : float
        Bootstrap standard error.
    ci : tuple of float
        Percentile confidence interval.
    p_value : float
        Bootstrap p-value.
    """
    if not np.isfinite(original_effect):
        return np.nan, (np.nan, np.nan), np.nan

    finite_mask = np.isfinite(boot_dist)
    n_valid = np.sum(finite_mask)
    n_total = len(boot_dist)

    if n_valid < n_total:
        n_nonfinite = n_total - n_valid
        warnings.warn(
            f"Dropping {n_nonfinite}/{n_total} non-finite bootstrap samples "
            f"in {context}. Bootstrap estimates based on remaining valid samples.",
            RuntimeWarning,
            stacklevel=3,
        )

    if n_valid < n_total * 0.5:
        warnings.warn(
            f"Too few valid bootstrap samples ({n_valid}/{n_total}) in {context}. "
            "Returning NaN for SE/CI/p-value to signal invalid inference.",
            RuntimeWarning,
            stacklevel=3,
        )
        return np.nan, (np.nan, np.nan), np.nan

    valid_dist = boot_dist[finite_mask]
    se = float(np.std(valid_dist, ddof=1))

    # Guard: if SE is not finite or zero, all inference fields must be NaN.
    if not np.isfinite(se) or se <= 0:
        warnings.warn(
            f"Bootstrap SE is non-finite or zero (n_valid={n_valid}) in {context}. "
            "Returning NaN for SE/CI/p-value.",
            RuntimeWarning,
            stacklevel=3,
        )
        return np.nan, (np.nan, np.nan), np.nan

    ci = compute_percentile_ci(valid_dist, alpha)
    p_value = compute_bootstrap_pvalue(original_effect, valid_dist, n_valid=len(valid_dist))
    return se, ci, p_value


def compute_effect_bootstrap_stats_batch(
    original_effects: np.ndarray,
    bootstrap_matrix: np.ndarray,
    alpha: float = 0.05,
) -> tuple:
    """
    Batch-compute bootstrap statistics for multiple effects at once.

    Parameters
    ----------
    original_effects : np.ndarray
        Array of original point estimates, shape (n_effects,).
    bootstrap_matrix : np.ndarray
        Bootstrap distributions, shape (n_bootstrap, n_effects).
    alpha : float, default=0.05
        Significance level.

    Returns
    -------
    ses : np.ndarray
        Bootstrap SEs for each effect.
    ci_lowers : np.ndarray
        Lower CI bounds for each effect.
    ci_uppers : np.ndarray
        Upper CI bounds for each effect.
    p_values : np.ndarray
        Bootstrap p-values for each effect.
    """
    n_bootstrap, n_effects = bootstrap_matrix.shape
    ses = np.full(n_effects, np.nan)
    ci_lowers = np.full(n_effects, np.nan)
    ci_uppers = np.full(n_effects, np.nan)
    p_values = np.full(n_effects, np.nan)

    # Check for non-finite original effects
    valid_effects = np.isfinite(original_effects)
    if not np.any(valid_effects):
        return ses, ci_lowers, ci_uppers, p_values

    # Count valid bootstrap samples per effect
    finite_mask = np.isfinite(bootstrap_matrix)  # (n_bootstrap, n_effects)
    n_valid = finite_mask.sum(axis=0)  # (n_effects,)

    # Determine which effects have enough valid samples
    enough_valid = (n_valid >= n_bootstrap * 0.5) & valid_effects

    if not np.any(enough_valid):
        n_insufficient = int(np.sum(valid_effects))
        if n_insufficient > 0:
            warnings.warn(
                f"{n_insufficient} effect(s) had too few valid bootstrap samples (<50%). "
                "Returning NaN for SE/CI/p-value.",
                RuntimeWarning,
                stacklevel=2,
            )
        return ses, ci_lowers, ci_uppers, p_values

    # Warn about subset with insufficient samples
    n_insufficient = int(np.sum(valid_effects & ~enough_valid))
    if n_insufficient > 0:
        warnings.warn(
            f"{n_insufficient} effect(s) had too few valid bootstrap samples (<50%). "
            "Returning NaN for SE/CI/p-value.",
            RuntimeWarning,
            stacklevel=2,
        )

    # For effects with all-finite bootstraps (common case), use vectorized ops
    all_finite = (n_valid == n_bootstrap) & enough_valid
    if np.any(all_finite):
        idx = np.where(all_finite)[0]
        sub = bootstrap_matrix[:, idx]

        # Vectorized SE: std across bootstrap dimension
        batch_ses = np.std(sub, axis=0, ddof=1)

        # Vectorized percentile CI
        lower_pct = alpha / 2 * 100
        upper_pct = (1 - alpha / 2) * 100
        batch_ci = np.percentile(sub, [lower_pct, upper_pct], axis=0)

        # Vectorized p-values
        batch_p = np.empty(len(idx))
        for j, eff_idx in enumerate(idx):
            eff = original_effects[eff_idx]
            if eff >= 0:
                batch_p[j] = np.mean(sub[:, j] <= 0)
            else:
                batch_p[j] = np.mean(sub[:, j] >= 0)
        batch_p = np.minimum(2 * batch_p, 1.0)
        batch_p = np.maximum(batch_p, 1 / (n_bootstrap + 1))

        # Guard: SE must be positive and finite
        se_valid = np.isfinite(batch_ses) & (batch_ses > 0)
        n_bad_se = int(np.sum(~se_valid))
        if n_bad_se > 0:
            warnings.warn(
                f"{n_bad_se} effect(s) had non-finite or zero bootstrap SE. "
                "Returning NaN for SE/CI/p-value.",
                RuntimeWarning,
                stacklevel=2,
            )
        ses[idx[se_valid]] = batch_ses[se_valid]
        ci_lowers[idx[se_valid]] = batch_ci[0][se_valid]
        ci_uppers[idx[se_valid]] = batch_ci[1][se_valid]
        p_values[idx[se_valid]] = batch_p[se_valid]

    # Handle effects with some non-finite bootstraps (rare) via scalar fallback
    partial_valid = enough_valid & ~all_finite
    if np.any(partial_valid):
        for j in np.where(partial_valid)[0]:
            se, ci, pv = compute_effect_bootstrap_stats(
                original_effects[j],
                bootstrap_matrix[:, j],
                alpha=alpha,
                context=f"effect {j}",
            )
            ses[j] = se
            ci_lowers[j] = ci[0]
            ci_uppers[j] = ci[1]
            p_values[j] = pv

    return ses, ci_lowers, ci_uppers, p_values


# ---------------------------------------------------------------------------
# Survey-aware bootstrap weight generators
# ---------------------------------------------------------------------------


def generate_survey_multiplier_weights_batch(
    n_bootstrap: int,
    resolved_survey: "ResolvedSurveyDesign",
    weight_type: str,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate PSU-level multiplier weights for survey-aware bootstrap.

    Within each stratum, weights are generated independently.  When FPC
    is present, weights are scaled by ``sqrt(1 - f_h)`` per stratum so
    the bootstrap variance matches the TSL variance.

    Parameters
    ----------
    n_bootstrap : int
        Number of bootstrap iterations.
    resolved_survey : ResolvedSurveyDesign
        Resolved survey design.
    weight_type : str
        Multiplier distribution: ``"rademacher"``, ``"mammen"``, or ``"webb"``.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    weights : np.ndarray
        Multiplier weights, shape ``(n_bootstrap, n_psu)``.
    psu_ids : np.ndarray
        Unique PSU identifiers aligned to columns of *weights*.
    """
    psu = resolved_survey.psu
    strata = resolved_survey.strata

    if resolved_survey.lonely_psu == "adjust":
        raise NotImplementedError(
            "lonely_psu='adjust' is not yet supported for survey-aware bootstrap. "
            "Use lonely_psu='remove' or 'certainty', or use analytical inference."
        )

    if psu is None:
        # Each observation is its own PSU
        n_psu = len(resolved_survey.weights)
        psu_ids = np.arange(n_psu)
    else:
        psu_ids = np.unique(psu)
        n_psu = len(psu_ids)

    if strata is None:
        # No stratification — generate a single block of weights
        if n_psu < 2:
            # Single PSU — variance unidentified (matches compute_survey_vcov)
            weights = np.zeros((n_bootstrap, n_psu), dtype=np.float64)
            return weights, psu_ids
        weights = generate_bootstrap_weights_batch(n_bootstrap, n_psu, weight_type, rng)
        # FPC scaling (unstratified)
        if resolved_survey.fpc is not None:
            if psu is not None:
                n_units_for_fpc = n_psu
            else:
                n_units_for_fpc = len(resolved_survey.weights)
            if resolved_survey.fpc[0] < n_units_for_fpc:
                raise ValueError(
                    f"FPC ({resolved_survey.fpc[0]}) is less than the number of PSUs "
                    f"({n_units_for_fpc}). FPC must be >= number of PSUs."
                )
            f = n_units_for_fpc / resolved_survey.fpc[0]
            if f < 1.0:
                weights = weights * np.sqrt(1.0 - f)
            else:
                weights = np.zeros_like(weights)
    else:
        # Stratified — generate independently within strata
        weights = np.empty((n_bootstrap, n_psu), dtype=np.float64)

        # Build PSU → column-index map
        psu_to_col = {int(p): i for i, p in enumerate(psu_ids)}

        unique_strata = np.unique(strata)
        for h in unique_strata:
            mask_h = strata == h

            if psu is not None:
                psus_in_h = np.unique(psu[mask_h])
            else:
                psus_in_h = np.where(mask_h)[0]

            n_h = len(psus_in_h)
            cols = np.array([psu_to_col[int(p)] for p in psus_in_h])

            if n_h < 2:
                # Lonely PSU — zero weight (matches remove/certainty behavior)
                weights[:, cols] = 0.0
                continue

            # Generate weights for this stratum
            stratum_weights = generate_bootstrap_weights_batch_numpy(
                n_bootstrap, n_h, weight_type, rng
            )

            # FPC scaling
            if resolved_survey.fpc is not None:
                N_h = resolved_survey.fpc[mask_h][0]
                if N_h < n_h:
                    raise ValueError(
                        f"FPC ({N_h}) is less than the number of PSUs "
                        f"({n_h}) in stratum {h}. FPC must be >= n_PSU."
                    )
                f_h = n_h / N_h
                if f_h < 1.0:
                    stratum_weights = stratum_weights * np.sqrt(1.0 - f_h)
                else:
                    stratum_weights = np.zeros_like(stratum_weights)

            weights[:, cols] = stratum_weights

    return weights, psu_ids


def generate_rao_wu_weights(
    resolved_survey: "ResolvedSurveyDesign",
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate one set of Rao-Wu (1988) rescaled observation weights.

    Within each stratum *h* with *n_h* PSUs, draw ``m_h`` PSUs with
    replacement and rescale observation weights by ``(n_h / m_h) * r_hi``
    where ``r_hi`` is the count of PSU *i* being selected.

    Without FPC: ``m_h = n_h - 1``.
    With FPC: ``m_h = max(1, round((1 - f_h) * (n_h - 1)))``
    (Rao, Wu & Yue 1992, Section 3).

    Parameters
    ----------
    resolved_survey : ResolvedSurveyDesign
        Resolved survey design.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Rescaled observation weights, shape ``(n_obs,)``.
    """
    n_obs = len(resolved_survey.weights)
    base_weights = resolved_survey.weights
    psu = resolved_survey.psu
    strata = resolved_survey.strata

    if resolved_survey.lonely_psu == "adjust":
        raise NotImplementedError(
            "lonely_psu='adjust' is not yet supported for survey-aware bootstrap. "
            "Use lonely_psu='remove' or 'certainty', or use analytical inference."
        )

    rescaled = np.zeros(n_obs, dtype=np.float64)

    if psu is None:
        obs_psu = np.arange(n_obs)
    else:
        obs_psu = psu

    if strata is None:
        strata_masks = [np.ones(n_obs, dtype=bool)]
    else:
        unique_strata = np.unique(strata)
        strata_masks = [strata == h for h in unique_strata]

    for mask_h in strata_masks:
        psu_h = obs_psu[mask_h]
        unique_psu_h = np.unique(psu_h)
        n_h = len(unique_psu_h)

        if n_h < 2:
            # Census / lonely PSU — keep original weights (zero variance)
            rescaled[mask_h] = base_weights[mask_h]
            continue

        # Compute resample size
        if resolved_survey.fpc is not None:
            N_h = resolved_survey.fpc[mask_h][0]
            if N_h < n_h:
                raise ValueError(
                    f"FPC ({N_h}) is less than the number of PSUs "
                    f"({n_h}). FPC must be >= number of PSUs."
                )
            f_h = n_h / N_h
            if f_h >= 1.0:
                # Census stratum — keep original weights (zero variance)
                rescaled[mask_h] = base_weights[mask_h]
                continue
            m_h = max(1, round((1.0 - f_h) * (n_h - 1)))
        else:
            m_h = n_h - 1

        # Draw m_h PSUs with replacement
        drawn_indices = rng.choice(n_h, size=m_h, replace=True)
        counts = np.bincount(drawn_indices, minlength=n_h)

        # Rescale factor per PSU: (n_h / m_h) * r_hi
        scale_per_psu = (n_h / m_h) * counts.astype(np.float64)

        # Map PSU → local index for vectorized application
        psu_to_local = {int(p): i for i, p in enumerate(unique_psu_h)}
        obs_in_h = np.where(mask_h)[0]
        local_indices = np.array([psu_to_local[int(obs_psu[idx])] for idx in obs_in_h])
        rescaled[obs_in_h] = base_weights[obs_in_h] * scale_per_psu[local_indices]

    return rescaled


def generate_rao_wu_weights_batch(
    n_bootstrap: int,
    resolved_survey: "ResolvedSurveyDesign",
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate multiple sets of Rao-Wu rescaled weights.

    Parameters
    ----------
    n_bootstrap : int
        Number of bootstrap iterations.
    resolved_survey : ResolvedSurveyDesign
        Resolved survey design.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Rescaled weights, shape ``(n_bootstrap, n_obs)``.
    """
    n_obs = len(resolved_survey.weights)
    result = np.empty((n_bootstrap, n_obs), dtype=np.float64)
    for b in range(n_bootstrap):
        result[b] = generate_rao_wu_weights(resolved_survey, rng)
    return result
