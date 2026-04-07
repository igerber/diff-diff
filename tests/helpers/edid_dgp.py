"""
Shared Compustat-style DGP for EfficientDiD tests.

Used by both test_efficient_did.py and test_efficient_did_validation.py
to avoid duplication. Based on Section 5.2 of Chen, Sant'Anna & Xie (2025).
"""

import numpy as np
import pandas as pd

# DGP parameters — treatment effect coefficients
ATT_COEFS = {5: 0.154, 8: 0.093}
N_PERIODS = 11


def make_compustat_dgp(
    n_units=400,
    n_periods=N_PERIODS,
    rho=0.0,
    seed=42,
    add_covariates=False,
    covariate_effect=0.5,
    confounding_strength=0.0,
):
    """Simplified Compustat-style DGP from Section 5.2.

    Groups: G=5 (~1/3), G=8 (~1/3), G=inf (~1/3).
    ATT(5,t) = 0.154*(t-4), ATT(8,t) = 0.093*(t-7).

    Parameters
    ----------
    add_covariates : bool
        If True, adds ``x1`` (continuous) and ``x2`` (binary) columns.
    covariate_effect : float
        Coefficient of ``x1`` on outcome (explains residual variance).
    confounding_strength : float
        ``x1`` also shifts the group-mean outcome differentially by group,
        creating confounding where parallel trends hold conditional on X
        but not unconditionally.  Set > 0 to test the DR path.
    """
    rng = np.random.default_rng(seed)
    n_t = n_periods

    n_g5 = n_units // 3
    n_g8 = n_units // 3
    ft = np.full(n_units, np.inf)
    ft[:n_g5] = 5
    ft[n_g5 : n_g5 + n_g8] = 8

    # Generate unit-level covariates (time-invariant)
    # When confounding_strength > 0, shift x1 distribution by group so
    # treated units have higher x1 on average (selection on observables).
    if confounding_strength > 0:
        x1 = np.empty(n_units)
        x1[:n_g5] = rng.normal(1.0, 1.0, n_g5)  # G=5: higher x1
        x1[n_g5 : n_g5 + n_g8] = rng.normal(0.5, 1.0, n_g8)  # G=8: moderate
        x1[n_g5 + n_g8 :] = rng.normal(0.0, 1.0, n_units - n_g5 - n_g8)  # G=inf: baseline
    else:
        x1 = rng.normal(0, 1, n_units)
    x2 = rng.binomial(1, 0.5, n_units)

    units = np.repeat(np.arange(n_units), n_t)
    times = np.tile(np.arange(1, n_t + 1), n_units)
    ft_col = np.repeat(ft, n_t)

    alpha_t = rng.normal(0, 0.1, n_t)
    eta_i = rng.normal(0, 0.5, n_units)
    unit_fe = np.repeat(eta_i, n_t)
    time_fe = np.tile(alpha_t, n_units)

    eps = np.zeros((n_units, n_t))
    eps[:, 0] = rng.normal(0, 0.3, n_units)
    for t in range(1, n_t):
        eps[:, t] = rho * eps[:, t - 1] + rng.normal(0, 0.3, n_units)
    eps_flat = eps.flatten()

    tau = np.zeros(len(units))
    for i in range(n_units):
        g = ft[i]
        if np.isinf(g):
            continue
        for t_idx in range(n_t):
            t = t_idx + 1
            if g == 5 and t >= 5:
                tau[i * n_t + t_idx] = ATT_COEFS[5] * (t - 4)
            elif g == 8 and t >= 8:
                tau[i * n_t + t_idx] = ATT_COEFS[8] * (t - 7)

    y = unit_fe + time_fe + tau + eps_flat

    if add_covariates:
        # Covariate effect on outcome (explains residual variance)
        x1_expanded = np.repeat(x1, n_t)
        y += covariate_effect * x1_expanded

        if confounding_strength > 0:
            # Confounding: x1-dependent time trend applied to ALL units.
            # Conditional on x1, all groups share the same trend → conditional PT holds.
            # Unconditional PT fails because treated/control have different x1 distributions.
            x1_expanded = np.repeat(x1, n_t)
            time_trend = np.tile(np.arange(n_t), n_units) / n_t
            y += confounding_strength * x1_expanded * time_trend

    result = pd.DataFrame({"unit": units, "time": times, "first_treat": ft_col, "y": y})

    if add_covariates:
        result["x1"] = np.repeat(x1, n_t)
        result["x2"] = np.repeat(x2, n_t)

    return result


def true_es_avg():
    """Derive ES_avg from DGP treatment effect parameters."""
    max_e = {g: N_PERIODS - g for g in ATT_COEFS}
    all_e = range(0, max(max_e.values()) + 1)
    es_values = []
    for e in all_e:
        contributing = [coef * (e + 1) for g, coef in ATT_COEFS.items() if e <= max_e[g]]
        if contributing:
            es_values.append(np.mean(contributing))
    return np.mean(es_values)


def true_overall_att():
    """Compute true overall_att using cohort-size weighting (library convention)."""
    effects = []
    for g, coef in ATT_COEFS.items():
        for t in range(g, N_PERIODS + 1):
            effects.append(coef * (t - g + 1))
    return np.mean(effects)
