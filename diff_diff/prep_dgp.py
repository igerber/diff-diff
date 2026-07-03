"""
Data generation utilities for difference-in-differences analysis.

This module provides functions to generate synthetic datasets for testing
and validating DiD estimators, including basic 2x2 DiD, staggered adoption,
factor model data, triple difference, and event study designs.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def generate_did_data(
    n_units: int = 100,
    n_periods: int = 4,
    treatment_effect: float = 5.0,
    treatment_fraction: float = 0.5,
    treatment_period: int = 2,
    unit_fe_sd: float = 2.0,
    time_trend: float = 0.5,
    noise_sd: float = 1.0,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate synthetic data for DiD analysis with known treatment effect.

    Creates a balanced panel dataset with realistic features including
    unit fixed effects, time trends, and a known treatment effect.

    Parameters
    ----------
    n_units : int, default=100
        Number of units in the panel.
    n_periods : int, default=4
        Number of time periods.
    treatment_effect : float, default=5.0
        True average treatment effect on the treated.
    treatment_fraction : float, default=0.5
        Fraction of units that receive treatment.
    treatment_period : int, default=2
        First post-treatment period (0-indexed). Periods >= this are post.
    unit_fe_sd : float, default=2.0
        Standard deviation of unit fixed effects.
    time_trend : float, default=0.5
        Linear time trend coefficient.
    noise_sd : float, default=1.0
        Standard deviation of idiosyncratic noise.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Synthetic panel data with columns:
        - unit: Unit identifier
        - period: Time period
        - treated: Treatment indicator (0/1)
        - post: Post-treatment indicator (0/1)
        - outcome: Outcome variable
        - true_effect: The true treatment effect (for validation)

    Examples
    --------
    Generate simple data for testing:

    >>> data = generate_did_data(n_units=50, n_periods=4, treatment_effect=3.0, seed=42)
    >>> len(data)
    200
    >>> data.columns.tolist()
    ['unit', 'period', 'treated', 'post', 'outcome', 'true_effect']

    Verify treatment effect recovery:

    >>> from diff_diff import DifferenceInDifferences
    >>> did = DifferenceInDifferences()
    >>> results = did.fit(data, outcome='outcome', treatment='treated', time='post')
    >>> abs(results.att - 3.0) < 1.0  # Close to true effect
    True
    """
    rng = np.random.default_rng(seed)

    # Determine treated units
    n_treated = int(n_units * treatment_fraction)
    treated_units = set(range(n_treated))

    # Generate unit fixed effects
    unit_fe = rng.normal(0, unit_fe_sd, n_units)

    # Build data
    records = []
    for unit in range(n_units):
        is_treated = unit in treated_units

        for period in range(n_periods):
            is_post = period >= treatment_period

            # Base outcome
            y = 10.0  # Baseline
            y += unit_fe[unit]  # Unit fixed effect
            y += time_trend * period  # Time trend

            # Treatment effect (only for treated units in post-period)
            effect = 0.0
            if is_treated and is_post:
                effect = treatment_effect
                y += effect

            # Add noise
            y += rng.normal(0, noise_sd)

            records.append(
                {
                    "unit": unit,
                    "period": period,
                    "treated": int(is_treated),
                    "post": int(is_post),
                    "outcome": y,
                    "true_effect": effect,
                }
            )

    return pd.DataFrame(records)


def generate_staggered_data(
    n_units: int = 100,
    n_periods: int = 10,
    cohort_periods: Optional[List[int]] = None,
    never_treated_frac: float = 0.3,
    treatment_effect: float = 2.0,
    dynamic_effects: bool = True,
    effect_growth: float = 0.1,
    unit_fe_sd: float = 2.0,
    time_trend: float = 0.1,
    noise_sd: float = 0.5,
    seed: Optional[int] = None,
    panel: bool = True,
) -> pd.DataFrame:
    """
    Generate synthetic data for staggered adoption DiD analysis.

    Creates panel data where different units receive treatment at different
    times (staggered rollout). Useful for testing CallawaySantAnna,
    SunAbraham, and other staggered DiD estimators.

    Parameters
    ----------
    n_units : int, default=100
        Total number of units in the panel.
    n_periods : int, default=10
        Number of time periods.
    cohort_periods : list of int, optional
        Periods when treatment cohorts are first treated.
        If None, defaults to [3, 5, 7] for a 10-period panel.
    never_treated_frac : float, default=0.3
        Fraction of units that are never treated (cohort 0).
    treatment_effect : float, default=2.0
        Base treatment effect at time of treatment.
    dynamic_effects : bool, default=True
        If True, treatment effects grow over time since treatment.
    effect_growth : float, default=0.1
        Per-period growth in treatment effect (if dynamic_effects=True).
        Effect at time t since treatment: effect * (1 + effect_growth * t).
    unit_fe_sd : float, default=2.0
        Standard deviation of unit fixed effects.
    time_trend : float, default=0.1
        Linear time trend coefficient.
    noise_sd : float, default=0.5
        Standard deviation of idiosyncratic noise.
    seed : int, optional
        Random seed for reproducibility.
    panel : bool, default=True
        If True (default), generate balanced panel data (same units across
        all periods). If False, generate repeated cross-section data where
        each period draws independent observations with globally unique IDs.

    Returns
    -------
    pd.DataFrame
        Synthetic staggered adoption data with columns:
        - unit: Unit identifier
        - period: Time period
        - outcome: Outcome variable
        - first_treat: First treatment period (0 = never treated)
        - treated: Binary indicator (1 if treated at this observation)
        - treat: Binary unit-level ever-treated indicator
        - true_effect: The true treatment effect for this observation

    Examples
    --------
    Generate staggered adoption data:

    >>> data = generate_staggered_data(n_units=100, n_periods=10, seed=42)
    >>> data['first_treat'].value_counts().sort_index()
    0     30
    3     24
    5     23
    7     23
    Name: first_treat, dtype: int64

    Use with Callaway-Sant'Anna estimator:

    >>> from diff_diff import CallawaySantAnna
    >>> cs = CallawaySantAnna()
    >>> results = cs.fit(data, outcome='outcome', unit='unit',
    ...                  time='period', first_treat='first_treat')
    >>> results.overall_att > 0
    True
    """
    rng = np.random.default_rng(seed)

    # Default cohort periods if not specified
    if cohort_periods is None:
        cohort_periods = [3, 5, 7] if n_periods >= 8 else [n_periods // 3, 2 * n_periods // 3]

    # Validate cohort periods
    for cp in cohort_periods:
        if cp < 1 or cp >= n_periods:
            raise ValueError(f"Cohort period {cp} must be between 1 and {n_periods - 1}")

    # Determine number of never-treated and treated units
    n_never = int(n_units * never_treated_frac)
    n_treated = n_units - n_never

    if not panel:
        # --- Repeated cross-section mode ---
        # Each period draws n_units independent observations with unique IDs.
        # Cohorts are assigned from the same distribution as panel.
        records = []
        for period in range(n_periods):
            # For each period, draw fresh cohort assignments
            ft_period = np.zeros(n_units, dtype=int)
            if n_treated > 0:
                cohort_assignments = rng.choice(len(cohort_periods), size=n_treated)
                ft_period[n_never:] = [cohort_periods[c] for c in cohort_assignments]

            # Unique unit IDs per period
            for i in range(n_units):
                uid = f"u{period}_{i}"
                unit_first_treat = ft_period[i]
                is_ever_treated = unit_first_treat > 0

                is_treated = is_ever_treated and period >= unit_first_treat

                # Outcome: unit_fe_proxy (drawn fresh) + time trend + treatment + noise
                unit_fe_proxy = rng.normal(0, unit_fe_sd)
                y = 10.0 + unit_fe_proxy + time_trend * period

                effect = 0.0
                if is_treated:
                    time_since_treatment = period - unit_first_treat
                    if dynamic_effects:
                        effect = treatment_effect * (1 + effect_growth * time_since_treatment)
                    else:
                        effect = treatment_effect
                    y += effect

                y += rng.normal(0, noise_sd)

                records.append(
                    {
                        "unit": uid,
                        "period": period,
                        "outcome": y,
                        "first_treat": unit_first_treat,
                        "treated": int(is_treated),
                        "treat": int(is_ever_treated),
                        "true_effect": effect,
                    }
                )

        return pd.DataFrame(records)

    # --- Panel mode (default) ---
    # Assign treatment cohorts
    first_treat = np.zeros(n_units, dtype=int)
    if n_treated > 0:
        cohort_assignments = rng.choice(len(cohort_periods), size=n_treated)
        first_treat[n_never:] = [cohort_periods[c] for c in cohort_assignments]

    # Generate unit fixed effects
    unit_fe = rng.normal(0, unit_fe_sd, n_units)

    # Build data
    records = []
    for unit in range(n_units):
        unit_first_treat = first_treat[unit]
        is_ever_treated = unit_first_treat > 0

        for period in range(n_periods):
            # Check if treated at this observation
            is_treated = is_ever_treated and period >= unit_first_treat

            # Base outcome: unit FE + time trend
            y = 10.0 + unit_fe[unit] + time_trend * period

            # Treatment effect
            effect = 0.0
            if is_treated:
                time_since_treatment = period - unit_first_treat
                if dynamic_effects:
                    effect = treatment_effect * (1 + effect_growth * time_since_treatment)
                else:
                    effect = treatment_effect
                y += effect

            # Add noise
            y += rng.normal(0, noise_sd)

            records.append(
                {
                    "unit": unit,
                    "period": period,
                    "outcome": y,
                    "first_treat": unit_first_treat,
                    "treated": int(is_treated),
                    "treat": int(is_ever_treated),
                    "true_effect": effect,
                }
            )

    return pd.DataFrame(records)


def generate_factor_data(
    n_units: int = 50,
    n_pre: int = 10,
    n_post: int = 5,
    n_treated: int = 10,
    n_factors: int = 2,
    treatment_effect: float = 2.0,
    factor_strength: float = 1.0,
    treated_loading_shift: float = 0.5,
    unit_fe_sd: float = 1.0,
    noise_sd: float = 0.5,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate synthetic panel data with interactive fixed effects (factor model).

    Creates data following the DGP:
    Y_it = mu + alpha_i + beta_t + Lambda_i'F_t + tau*D_it + eps_it

    where Lambda_i'F_t is the interactive fixed effects component. Useful for
    testing TROP (Triply Robust Panel) and comparing with SyntheticDiD.

    Parameters
    ----------
    n_units : int, default=50
        Total number of units in the panel.
    n_pre : int, default=10
        Number of pre-treatment periods.
    n_post : int, default=5
        Number of post-treatment periods.
    n_treated : int, default=10
        Number of treated units (assigned to first n_treated unit IDs).
    n_factors : int, default=2
        Number of latent factors in the interactive fixed effects.
    treatment_effect : float, default=2.0
        True average treatment effect on the treated.
    factor_strength : float, default=1.0
        Scaling factor for interactive fixed effects.
    treated_loading_shift : float, default=0.5
        Shift in factor loadings for treated units (creates confounding).
    unit_fe_sd : float, default=1.0
        Standard deviation of unit fixed effects.
    noise_sd : float, default=0.5
        Standard deviation of idiosyncratic noise.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Synthetic factor model data with columns:
        - unit: Unit identifier
        - period: Time period
        - outcome: Outcome variable
        - treated: Binary indicator (1 if treated at this observation)
        - treat: Binary unit-level ever-treated indicator
        - true_effect: The true treatment effect for this observation

    Examples
    --------
    Generate data with factor structure:

    >>> data = generate_factor_data(n_units=50, n_factors=2, seed=42)
    >>> data.shape
    (750, 6)

    Use with TROP estimator:

    >>> from diff_diff import TROP
    >>> trop = TROP(n_bootstrap=50, seed=42)
    >>> results = trop.fit(data, outcome='outcome', treatment='treated',
    ...                    unit='unit', time='period',
    ...                    post_periods=list(range(10, 15)))

    Notes
    -----
    The treated units have systematically different factor loadings
    (shifted by `treated_loading_shift`), which creates confounding
    that standard DiD cannot address but TROP can handle.
    """
    rng = np.random.default_rng(seed)

    n_periods = n_pre + n_post

    if n_treated > n_units:
        raise ValueError(f"n_treated ({n_treated}) cannot exceed n_units ({n_units})")
    if n_treated < 1:
        raise ValueError("n_treated must be at least 1")

    # Generate factors F: (n_periods, n_factors)
    F = rng.normal(0, 1, (n_periods, n_factors))

    # Generate loadings Lambda: (n_factors, n_units)
    # Treated units have shifted loadings (creates confounding)
    Lambda = rng.normal(0, 1, (n_factors, n_units))
    Lambda[:, :n_treated] += treated_loading_shift

    # Unit fixed effects (treated units have higher baseline)
    alpha = rng.normal(0, unit_fe_sd, n_units)
    alpha[:n_treated] += 1.0

    # Time fixed effects (linear trend)
    beta = np.linspace(0, 2, n_periods)

    # Generate outcomes
    records = []
    for i in range(n_units):
        is_ever_treated = i < n_treated

        for t in range(n_periods):
            post = t >= n_pre

            # Base outcome
            y = 10.0 + alpha[i] + beta[t]

            # Interactive fixed effects: Lambda_i' F_t
            y += factor_strength * (Lambda[:, i] @ F[t, :])

            # Treatment effect
            effect = 0.0
            if is_ever_treated and post:
                effect = treatment_effect
                y += effect

            # Add noise
            y += rng.normal(0, noise_sd)

            records.append(
                {
                    "unit": i,
                    "period": t,
                    "outcome": y,
                    "treated": int(is_ever_treated and post),
                    "treat": int(is_ever_treated),
                    "true_effect": effect,
                }
            )

    return pd.DataFrame(records)


def generate_synthetic_control_data(
    n_donors: int = 20,
    n_pre: int = 24,
    n_post: int = 6,
    n_factors: int = 3,
    n_predictors: int = 3,
    treatment_effect: float = 5.0,
    effect_type: str = "ramp",
    effect_growth: float = 1.0,
    factor_strength: float = 1.0,
    factor_persistence: float = 0.9,
    n_convex_donors: int = 4,
    baseline: float = 50.0,
    unit_baseline_sd: float = 5.0,
    time_trend: float = 0.5,
    predictor_noise_sd: float = 0.5,
    noise_sd: float = 1.0,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate a single-treated-unit panel for synthetic control (ADH) demos.

    Creates a factor-model panel with ONE treated unit (``unit == 0``) and
    ``n_donors`` never-treated donors, designed so the treated unit's underlying
    (noiseless) trajectory lies in the span of the donor trajectories — so a synthetic
    control can reproduce it closely. The DGP:

        Y_it = b_i + beta_t + factor_strength * (Lambda_i . F_t) + tau_it + eps_it

    where ``F_t`` is a persistent (AR(1)) latent factor path, ``Lambda_i`` are
    nonnegative factor loadings, ``b_i`` is a unit baseline, ``beta_t`` is a common
    calendar time effect, ``tau_it`` is the treatment effect (treated unit, post
    periods only), and ``eps_it ~ N(0, noise_sd)``.

    The treated unit's loadings and baseline are an exact convex combination of
    ``n_convex_donors`` donors (Dirichlet weights), so in the **noiseless** limit the
    treated unit's outcome path is itself that convex combination of donor paths — it
    lies in the donor convex hull and a synthetic control reproduces it exactly. With
    nonzero ``noise_sd`` / ``predictor_noise_sd`` the observed data is only
    *approximately* in-hull, so a fitted synthetic control achieves a small (not
    exactly zero) pre-period RMSPE. Note that the generating weights are NOT identified
    by the estimator: many donor combinations reproduce the treated pre-period path
    equally well, so a fitted synthetic control recovers the counterfactual outcome path
    (and ATT), not this specific weight vector.

    Unlike :func:`generate_factor_data` (which *shifts* treated loadings to create
    confounding for TROP), this generator keeps the treated unit reproducible from
    donors, which is what classic synthetic control requires.

    Parameters
    ----------
    n_donors : int, default=20
        Number of never-treated donor units. The in-space placebo p-value floor is
        ``1 / (n_donors + 1)``; larger pools give finer permutation p-values.
    n_pre : int, default=24
        Number of pre-treatment periods.
    n_post : int, default=6
        Number of post-treatment periods.
    n_factors : int, default=3
        Number of latent factors driving the common time structure.
    n_predictors : int, default=3
        Number of time-invariant predictor covariates (each a noisy linear map of
        the unit's loading vector). Pass these as ``predictors=`` to
        :class:`~diff_diff.SyntheticControl` to exercise the V-search; otherwise
        the fit defaults to pre-period outcomes and these columns are unused.
    treatment_effect : float, default=5.0
        Base treatment effect on the treated unit in post periods. Under
        ``effect_type="ramp"`` this is the first post-period effect.
    effect_type : str, default="ramp"
        ``"ramp"`` (effect grows by ``effect_growth`` each post period) or
        ``"constant"`` (effect fixed at ``treatment_effect``).
    effect_growth : float, default=1.0
        Per-post-period increment under ``effect_type="ramp"`` (ignored otherwise).
    factor_strength : float, default=1.0
        Scaling on the interactive (loading . factor) component.
    factor_persistence : float, default=0.9
        AR(1) coefficient on the latent factors in ``[0, 1]``: ``0`` gives i.i.d.
        factors, ``1`` a random walk. Serial dependence is what makes the
        moving-block conformal scheme meaningful.
    n_convex_donors : int, default=4
        Number of donors whose convex combination defines the treated unit (must be
        in ``[1, n_donors]``).
    baseline : float, default=50.0
        Mean unit baseline (e.g. a per-capita outcome level).
    unit_baseline_sd : float, default=5.0
        Standard deviation of donor baselines.
    time_trend : float, default=0.5
        Slope of the common calendar time effect ``beta_t = time_trend * t`` (a
        common trend is matched by any convex combination, so it preserves in-hull
        existence). Set to ``0.0`` for no trend.
    predictor_noise_sd : float, default=0.5
        Noise added to each predictor covariate.
    noise_sd : float, default=1.0
        Standard deviation of idiosyncratic outcome noise.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Long-format balanced panel with columns:

        - ``unit``: unit identifier (``0`` is the treated unit; ``1..n_donors`` donors)
        - ``period``: time period (``0..n_pre-1`` pre, ``n_pre..`` post)
        - ``outcome``: outcome variable
        - ``treatment``: absorbing 0/1 indicator (1 only for the treated unit in
          post periods) — :meth:`SyntheticControl.fit` infers the treated unit and
          post periods from this column
        - ``treat``: unit-level ever-treated flag
        - ``true_effect``: the true treatment effect for this observation
        - ``x1 .. x{n_predictors}``: predictor covariates

    Examples
    --------
    Generate a ramped-effect panel and fit a synthetic control:

    >>> from diff_diff import generate_synthetic_control_data, SyntheticControl
    >>> data = generate_synthetic_control_data(seed=42)
    >>> res = SyntheticControl(seed=42).fit(
    ...     data, outcome="outcome", treatment="treatment",
    ...     unit="unit", time="period", predictors=["x1", "x2", "x3"])
    >>> round(res.att, 1) > 0  # post-period gap recovers the injected effect
    True
    """
    if n_donors < 2:
        raise ValueError(f"n_donors ({n_donors}) must be at least 2")
    if n_pre < 1:
        raise ValueError(f"n_pre ({n_pre}) must be at least 1")
    if n_post < 1:
        raise ValueError(f"n_post ({n_post}) must be at least 1")
    if n_factors < 1:
        raise ValueError(f"n_factors ({n_factors}) must be at least 1")
    if n_predictors < 0:
        raise ValueError(f"n_predictors ({n_predictors}) must be non-negative")
    if n_convex_donors < 1 or n_convex_donors > n_donors:
        raise ValueError(f"n_convex_donors ({n_convex_donors}) must be in [1, n_donors={n_donors}]")
    if effect_type not in ("ramp", "constant"):
        raise ValueError(f"effect_type ({effect_type!r}) must be 'ramp' or 'constant'")
    if not (0.0 <= factor_persistence <= 1.0):
        raise ValueError(f"factor_persistence ({factor_persistence}) must be in [0, 1]")

    rng = np.random.default_rng(seed)

    n_periods = n_pre + n_post
    n_units = n_donors + 1  # unit 0 is treated, 1..n_donors are donors

    # Persistent latent factors F: (n_periods, n_factors). AR(1): rho=1 -> random
    # walk, rho=0 -> i.i.d. The synthetic control removes this common structure, so
    # the post-fit residuals stay stationary regardless of persistence.
    F = np.empty((n_periods, n_factors))
    F[0] = rng.normal(0, 1, n_factors)
    for t in range(1, n_periods):
        F[t] = factor_persistence * F[t - 1] + rng.normal(0, 1, n_factors)

    # Donor loadings (nonnegative so a convex combination is well-defined) and
    # baselines.
    lambda_donor = np.abs(rng.normal(0, 1, (n_factors, n_donors)))
    b_donor = rng.normal(baseline, unit_baseline_sd, n_donors)

    # Treated unit = exact convex combination of n_convex_donors donors (in-hull).
    convex_idx = rng.choice(n_donors, size=n_convex_donors, replace=False)
    w_star = np.zeros(n_donors)
    w_star[convex_idx] = rng.dirichlet(np.ones(n_convex_donors))
    lambda_treated = lambda_donor @ w_star
    b_treated = float(b_donor @ w_star)

    # Stack: column 0 = treated, columns 1.. = donors.
    loadings = np.column_stack([lambda_treated, lambda_donor])  # (n_factors, n_units)
    baselines = np.concatenate([[b_treated], b_donor])  # (n_units,)

    # Time-invariant predictors: each PRIMARILY proxies one distinct latent factor
    # (diagonal-dominant map) plus a small cross-factor mix and noise. Distinct
    # proxies give each predictor complementary signal, so the V-search spreads
    # weight across them instead of leaning on a single redundant column.
    pred_map = 0.25 * rng.normal(0, 1, (n_predictors, n_factors))
    for h in range(n_predictors):
        pred_map[h, h % n_factors] += 1.0
    predictors = pred_map @ loadings + rng.normal(
        0, predictor_noise_sd, (n_predictors, n_units)
    )  # (n_predictors, n_units)

    # Common calendar time effect (matched by any convex combination since sum w=1).
    beta = time_trend * np.arange(n_periods)

    records = []
    for i in range(n_units):
        is_treated_unit = i == 0
        for t in range(n_periods):
            post = t >= n_pre
            y = baselines[i] + beta[t] + factor_strength * (loadings[:, i] @ F[t])

            effect = 0.0
            if is_treated_unit and post:
                k = t - n_pre  # post-period index, 0-based
                if effect_type == "ramp":
                    effect = treatment_effect + effect_growth * k
                else:
                    effect = treatment_effect
                y += effect

            y += rng.normal(0, noise_sd)

            row = {
                "unit": i,
                "period": t,
                "outcome": y,
                "treatment": int(is_treated_unit and post),
                "treat": int(is_treated_unit),
                "true_effect": effect,
            }
            for h in range(n_predictors):
                row[f"x{h + 1}"] = predictors[h, i]
            records.append(row)

    return pd.DataFrame(records)


def generate_ddd_data(
    n_per_cell: int = 100,
    treatment_effect: float = 2.0,
    group_effect: float = 2.0,
    partition_effect: float = 1.0,
    time_effect: float = 0.5,
    noise_sd: float = 1.0,
    add_covariates: bool = False,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate synthetic data for Triple Difference (DDD) analysis.

    Creates data following the DGP:
    Y = mu + G + P + T + G*P + G*T + P*T + tau*G*P*T + eps

    where G=group, P=partition, T=time. The treatment effect (tau) only
    applies to units that are in the treated group (G=1), eligible
    partition (P=1), and post-treatment period (T=1).

    Parameters
    ----------
    n_per_cell : int, default=100
        Number of observations per cell (8 cells total: 2x2x2).
    treatment_effect : float, default=2.0
        True average treatment effect on the treated (G=1, P=1, T=1).
    group_effect : float, default=2.0
        Main effect of being in treated group.
    partition_effect : float, default=1.0
        Main effect of being in eligible partition.
    time_effect : float, default=0.5
        Main effect of post-treatment period.
    noise_sd : float, default=1.0
        Standard deviation of idiosyncratic noise.
    add_covariates : bool, default=False
        If True, adds age and education covariates that affect outcome.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Synthetic DDD data with columns:
        - outcome: Outcome variable
        - group: Group indicator (0=control, 1=treated)
        - partition: Partition indicator (0=ineligible, 1=eligible)
        - time: Time indicator (0=pre, 1=post)
        - unit_id: Unique unit identifier
        - true_effect: The true treatment effect for this observation
        - age: Age covariate (if add_covariates=True)
        - education: Education covariate (if add_covariates=True)

    Examples
    --------
    Generate DDD data:

    >>> data = generate_ddd_data(n_per_cell=100, treatment_effect=3.0, seed=42)
    >>> data.shape
    (800, 6)
    >>> data.groupby(['group', 'partition', 'time']).size()
    group  partition  time
    0      0          0       100
                      1       100
           1          0       100
                      1       100
    1      0          0       100
                      1       100
           1          0       100
                      1       100
    dtype: int64

    Use with TripleDifference estimator:

    >>> from diff_diff import TripleDifference
    >>> ddd = TripleDifference()
    >>> results = ddd.fit(data, outcome='outcome', group='group',
    ...                   partition='partition', time='time')
    >>> abs(results.att - 3.0) < 1.0
    True
    """
    rng = np.random.default_rng(seed)

    records = []
    unit_id = 0

    for g in [0, 1]:  # group (0=control state, 1=treated state)
        for p in [0, 1]:  # partition (0=ineligible, 1=eligible)
            for t in [0, 1]:  # time (0=pre, 1=post)
                for _ in range(n_per_cell):
                    # Base outcome with main effects
                    y = 50 + group_effect * g + partition_effect * p + time_effect * t

                    # Second-order interactions (non-treatment)
                    y += 1.5 * g * p  # group-partition interaction
                    y += 1.0 * g * t  # group-time interaction (diff trends)
                    y += 0.5 * p * t  # partition-time interaction

                    # Treatment effect: ONLY for G=1, P=1, T=1
                    effect = 0.0
                    if g == 1 and p == 1 and t == 1:
                        effect = treatment_effect
                        y += effect

                    # Covariates (always generated for consistency)
                    age = rng.normal(40, 10)
                    education = rng.choice([12, 14, 16, 18], p=[0.3, 0.3, 0.25, 0.15])

                    if add_covariates:
                        y += 0.1 * age + 0.5 * education

                    # Add noise
                    y += rng.normal(0, noise_sd)

                    record = {
                        "outcome": y,
                        "group": g,
                        "partition": p,
                        "time": t,
                        "unit_id": unit_id,
                        "true_effect": effect,
                    }

                    if add_covariates:
                        record["age"] = age
                        record["education"] = education

                    records.append(record)
                    unit_id += 1

    return pd.DataFrame(records)


def generate_panel_data(
    n_units: int = 100,
    n_periods: int = 8,
    treatment_period: int = 4,
    treatment_fraction: float = 0.5,
    treatment_effect: float = 5.0,
    parallel_trends: bool = True,
    trend_violation: float = 1.0,
    unit_fe_sd: float = 2.0,
    noise_sd: float = 0.5,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate synthetic panel data for parallel trends testing.

    Creates panel data with optional violation of parallel trends, useful
    for testing parallel trends diagnostics, placebo tests, and sensitivity
    analysis methods.

    Parameters
    ----------
    n_units : int, default=100
        Total number of units in the panel.
    n_periods : int, default=8
        Number of time periods.
    treatment_period : int, default=4
        First post-treatment period (0-indexed).
    treatment_fraction : float, default=0.5
        Fraction of units that receive treatment.
    treatment_effect : float, default=5.0
        True average treatment effect on the treated.
    parallel_trends : bool, default=True
        If True, treated and control groups have parallel pre-treatment trends.
        If False, treated group has a steeper pre-treatment trend.
    trend_violation : float, default=1.0
        Size of the differential trend for treated group when parallel_trends=False.
        Treated units have trend = common_trend + trend_violation.
    unit_fe_sd : float, default=2.0
        Standard deviation of unit fixed effects.
    noise_sd : float, default=0.5
        Standard deviation of idiosyncratic noise.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Synthetic panel data with columns:
        - unit: Unit identifier
        - period: Time period
        - treated: Binary unit-level treatment indicator
        - post: Binary post-treatment indicator
        - outcome: Outcome variable
        - true_effect: The true treatment effect for this observation

    Examples
    --------
    Generate data with parallel trends:

    >>> data_parallel = generate_panel_data(parallel_trends=True, seed=42)
    >>> from diff_diff.utils import check_parallel_trends
    >>> result = check_parallel_trends(data_parallel, outcome='outcome',
    ...                                time='period', treatment_group='treated',
    ...                                pre_periods=[0, 1, 2, 3])
    >>> result['parallel_trends_plausible']
    True

    Generate data with trend violation:

    >>> data_violation = generate_panel_data(parallel_trends=False, seed=42)
    >>> result = check_parallel_trends(data_violation, outcome='outcome',
    ...                                time='period', treatment_group='treated',
    ...                                pre_periods=[0, 1, 2, 3])
    >>> result['parallel_trends_plausible']
    False
    """
    rng = np.random.default_rng(seed)

    if treatment_period < 1:
        raise ValueError("treatment_period must be at least 1")
    if treatment_period >= n_periods:
        raise ValueError(f"treatment_period must be less than n_periods ({n_periods})")

    n_treated = int(n_units * treatment_fraction)

    records = []
    for unit in range(n_units):
        is_treated = unit < n_treated
        unit_fe = rng.normal(0, unit_fe_sd)

        for period in range(n_periods):
            post = period >= treatment_period

            # Base time effect (common trend)
            if parallel_trends:
                time_effect = period * 1.0
            else:
                # Different trends: treated has steeper pre-treatment trend
                if is_treated:
                    time_effect = period * (1.0 + trend_violation)
                else:
                    time_effect = period * 1.0

            y = 10.0 + unit_fe + time_effect

            # Treatment effect (only for treated in post-period)
            effect = 0.0
            if is_treated and post:
                effect = treatment_effect
                y += effect

            # Add noise
            y += rng.normal(0, noise_sd)

            records.append(
                {
                    "unit": unit,
                    "period": period,
                    "treated": int(is_treated),
                    "post": int(post),
                    "outcome": y,
                    "true_effect": effect,
                }
            )

    return pd.DataFrame(records)


def generate_event_study_data(
    n_units: int = 300,
    n_pre: int = 5,
    n_post: int = 5,
    treatment_fraction: float = 0.5,
    treatment_effect: float = 5.0,
    unit_fe_sd: float = 2.0,
    noise_sd: float = 2.0,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate synthetic data for event study analysis.

    Creates panel data with simultaneous treatment at period n_pre.
    Useful for testing MultiPeriodDiD, pre-trends power analysis,
    and HonestDiD sensitivity analysis.

    Parameters
    ----------
    n_units : int, default=300
        Total number of units in the panel.
    n_pre : int, default=5
        Number of pre-treatment periods.
    n_post : int, default=5
        Number of post-treatment periods.
    treatment_fraction : float, default=0.5
        Fraction of units that receive treatment.
    treatment_effect : float, default=5.0
        True average treatment effect on the treated.
    unit_fe_sd : float, default=2.0
        Standard deviation of unit fixed effects.
    noise_sd : float, default=2.0
        Standard deviation of idiosyncratic noise.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Synthetic event study data with columns:
        - unit: Unit identifier
        - period: Time period
        - treated: Binary unit-level treatment indicator
        - post: Binary post-treatment indicator
        - outcome: Outcome variable
        - event_time: Time relative to treatment (negative=pre, 0+=post)
        - true_effect: The true treatment effect for this observation

    Examples
    --------
    Generate event study data:

    >>> data = generate_event_study_data(n_units=300, n_pre=5, n_post=5, seed=42)
    >>> data['event_time'].unique()
    array([-5, -4, -3, -2, -1,  0,  1,  2,  3,  4])

    Use with MultiPeriodDiD:

    >>> from diff_diff import MultiPeriodDiD
    >>> mp_did = MultiPeriodDiD()
    >>> results = mp_did.fit(data, outcome='outcome', treatment='treated',
    ...                      time='period', post_periods=[5, 6, 7, 8, 9])

    Notes
    -----
    The event_time column is relative to treatment:
    - Negative values: pre-treatment periods
    - 0: first post-treatment period
    - Positive values: subsequent post-treatment periods
    """
    rng = np.random.default_rng(seed)

    n_periods = n_pre + n_post
    treatment_period = n_pre
    n_treated = int(n_units * treatment_fraction)

    records = []
    for unit in range(n_units):
        is_treated = unit < n_treated
        unit_fe = rng.normal(0, unit_fe_sd)

        for period in range(n_periods):
            post = period >= treatment_period
            event_time = period - treatment_period

            # Common time trend
            time_effect = period * 0.5

            y = 10.0 + unit_fe + time_effect

            # Treatment effect (only for treated in post-period)
            effect = 0.0
            if is_treated and post:
                effect = treatment_effect
                y += effect

            # Add noise
            y += rng.normal(0, noise_sd)

            records.append(
                {
                    "unit": unit,
                    "period": period,
                    "treated": int(is_treated),
                    "post": int(post),
                    "outcome": y,
                    "event_time": event_time,
                    "true_effect": effect,
                }
            )

    return pd.DataFrame(records)


def generate_continuous_did_data(
    n_units: int = 500,
    n_periods: int = 4,
    cohort_periods: Optional[List[int]] = None,
    never_treated_frac: float = 0.3,
    dose_distribution: str = "lognormal",
    dose_params: Optional[Dict] = None,
    att_function: str = "linear",
    att_slope: float = 2.0,
    att_intercept: float = 1.0,
    unit_fe_sd: float = 2.0,
    time_trend: float = 0.5,
    noise_sd: float = 1.0,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate synthetic data for continuous DiD analysis with known dose-response.

    Creates a balanced panel with continuous treatment doses and known ATT(d)
    function, satisfying strong parallel trends by construction.

    Parameters
    ----------
    n_units : int, default=500
        Number of units in the panel.
    n_periods : int, default=4
        Number of time periods (1-indexed).
    cohort_periods : list of int, optional
        Treatment cohort periods. Default: ``[2]`` (single cohort).
    never_treated_frac : float, default=0.3
        Fraction of units that are never-treated.
    dose_distribution : str, default="lognormal"
        Distribution for dose: ``"lognormal"``, ``"uniform"``, ``"exponential"``.
    dose_params : dict, optional
        Distribution-specific parameters. Defaults:
        lognormal: ``{"mean": 0.5, "sigma": 0.5}``
        uniform: ``{"low": 0.5, "high": 5.0}``
        exponential: ``{"scale": 2.0}``
    att_function : str, default="linear"
        Functional form of ATT(d): ``"linear"``, ``"quadratic"``, ``"log"``.
    att_slope : float, default=2.0
        Slope parameter for ATT function.
    att_intercept : float, default=1.0
        Intercept parameter for ATT function.
    unit_fe_sd : float, default=2.0
        Standard deviation of unit fixed effects.
    time_trend : float, default=0.5
        Linear time trend coefficient.
    noise_sd : float, default=1.0
        Standard deviation of idiosyncratic noise.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Panel data with columns: ``unit``, ``period``, ``outcome``,
        ``first_treat``, ``dose``, ``true_att``.
    """
    rng = np.random.default_rng(seed)

    if cohort_periods is None:
        cohort_periods = [2]

    # Assign units to cohorts
    n_never = int(n_units * never_treated_frac)
    n_treated_total = n_units - n_never
    n_per_cohort = n_treated_total // len(cohort_periods)

    cohort_assignments = np.zeros(n_units, dtype=int)
    idx = 0
    for i, g in enumerate(cohort_periods):
        n_this = n_per_cohort if i < len(cohort_periods) - 1 else n_treated_total - idx
        cohort_assignments[n_never + idx : n_never + idx + n_this] = g
        idx += n_this

    # Generate doses
    default_params = {
        "lognormal": {"mean": 0.5, "sigma": 0.5},
        "uniform": {"low": 0.5, "high": 5.0},
        "exponential": {"scale": 2.0},
    }
    params = dose_params or default_params.get(dose_distribution, {})

    dose_per_unit = np.zeros(n_units)
    treated_mask = cohort_assignments > 0
    n_treated_actual = int(np.sum(treated_mask))

    if dose_distribution == "lognormal":
        dose_per_unit[treated_mask] = rng.lognormal(
            mean=params.get("mean", 0.5),
            sigma=params.get("sigma", 0.5),
            size=n_treated_actual,
        )
    elif dose_distribution == "uniform":
        dose_per_unit[treated_mask] = rng.uniform(
            low=params.get("low", 0.5),
            high=params.get("high", 5.0),
            size=n_treated_actual,
        )
    elif dose_distribution == "exponential":
        dose_per_unit[treated_mask] = rng.exponential(
            scale=params.get("scale", 2.0),
            size=n_treated_actual,
        )
    else:
        raise ValueError(
            f"dose_distribution must be 'lognormal', 'uniform', or 'exponential', "
            f"got '{dose_distribution}'"
        )

    # ATT function
    def _att_func(d):
        if att_function == "linear":
            return att_intercept + att_slope * d
        elif att_function == "quadratic":
            return att_intercept + att_slope * d**2
        elif att_function == "log":
            return att_intercept + att_slope * np.log1p(d)
        else:
            raise ValueError(
                f"att_function must be 'linear', 'quadratic', or 'log', " f"got '{att_function}'"
            )

    # Unit fixed effects
    unit_fe = rng.normal(0, unit_fe_sd, size=n_units)

    # Build panel
    periods = np.arange(1, n_periods + 1)
    records = []
    for i in range(n_units):
        g_i = cohort_assignments[i]
        d_i = dose_per_unit[i]
        for t in periods:
            # Potential outcome without treatment
            y0 = unit_fe[i] + time_trend * t + rng.normal(0, noise_sd)
            # Treatment effect
            if g_i > 0 and t >= g_i:
                att_d = _att_func(d_i)
            else:
                att_d = 0.0

            records.append(
                {
                    "unit": i,
                    "period": int(t),
                    "outcome": y0 + att_d,
                    "first_treat": int(g_i) if g_i > 0 else 0,
                    "dose": d_i,
                    "true_att": att_d,
                }
            )

    return pd.DataFrame(records)


def generate_staggered_ddd_data(
    n_units: int = 200,
    n_periods: int = 8,
    cohort_periods: Optional[List[int]] = None,
    never_enabled_frac: float = 0.25,
    eligibility_frac: float = 0.5,
    treatment_effect: float = 3.0,
    dynamic_effects: bool = False,
    effect_growth: float = 0.1,
    eligibility_trend: float = 0.3,
    noise_sd: float = 0.5,
    add_covariates: bool = False,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate synthetic data for staggered triple difference (DDD) analysis.

    Creates a balanced panel with staggered enabling times and a binary
    eligibility dimension. Treatment occurs when a unit is both enabled
    (t >= S_i) and eligible (Q_i = 1). DDD-CPT holds by construction.

    Parameters
    ----------
    n_units : int, default=200
        Number of units.
    n_periods : int, default=8
        Number of time periods (1-indexed).
    cohort_periods : list of int, optional
        Enabling periods. Default: [4, 6].
    never_enabled_frac : float, default=0.25
        Fraction of never-enabled units.
    eligibility_frac : float, default=0.5
        Fraction of eligible units (Q=1) within each cohort.
    treatment_effect : float, default=3.0
        True ATT for treated units.
    dynamic_effects : bool, default=False
        If True, effects grow over time since enabling.
    effect_growth : float, default=0.1
        Per-period effect growth rate when dynamic_effects=True.
    eligibility_trend : float, default=0.3
        Differential time trend for eligible vs ineligible units.
        Same across all enabling groups (preserves DDD-CPT).
    noise_sd : float, default=0.5
        Standard deviation of idiosyncratic noise.
    add_covariates : bool, default=False
        If True, add covariates x1 (continuous) and x2 (binary).
    seed : int, optional
        Random seed.

    Returns
    -------
    pd.DataFrame
        Columns: unit, period, outcome, first_treat, eligibility, treated,
        true_effect. Also x1, x2 if add_covariates=True.
    """
    rng = np.random.default_rng(seed)

    if cohort_periods is None:
        cohort_periods = [4, 6]

    # Assign units to cohorts
    n_never = int(n_units * never_enabled_frac)
    n_treated_total = n_units - n_never
    n_per_cohort = n_treated_total // len(cohort_periods)

    unit_cohort = np.zeros(n_units, dtype=float)
    idx = n_never
    for i, g in enumerate(cohort_periods):
        n_g = n_per_cohort if i < len(cohort_periods) - 1 else n_treated_total - idx + n_never
        unit_cohort[idx : idx + n_g] = g
        idx += n_g

    # Assign eligibility (within each cohort, fraction eligible)
    unit_elig = np.zeros(n_units, dtype=int)
    for g_val in [0.0] + [float(g) for g in cohort_periods]:
        mask = unit_cohort == g_val
        n_g = int(np.sum(mask))
        if n_g == 0:
            continue
        n_eligible = max(1, min(int(n_g * eligibility_frac), n_g))
        indices = np.where(mask)[0]
        eligible_idx = rng.choice(indices, size=n_eligible, replace=False)
        unit_elig[eligible_idx] = 1

    # Unit fixed effects
    unit_fe = rng.normal(0, 2.0, size=n_units)

    # Covariates
    x1 = rng.normal(0, 1, size=n_units) if add_covariates else None
    x2 = rng.choice([0, 1], size=n_units) if add_covariates else None

    # Generate panel
    records = []
    for i in range(n_units):
        g_i = unit_cohort[i]
        q_i = unit_elig[i]
        for t in range(1, n_periods + 1):
            # Base: unit FE + time trend + eligibility-time interaction
            gamma_t = 0.1 * t
            y = unit_fe[i] + gamma_t + 1.0 * q_i + eligibility_trend * q_i * gamma_t

            if add_covariates:
                y += 0.5 * x1[i] + 0.3 * x2[i]

            # Treatment effect: enabled AND eligible
            treated = int(g_i > 0 and t >= g_i and q_i == 1)
            true_eff = 0.0
            if treated:
                true_eff = treatment_effect
                if dynamic_effects:
                    true_eff *= 1 + effect_growth * (t - g_i)
                y += true_eff

            y += rng.normal(0, noise_sd)

            row = {
                "unit": i,
                "period": t,
                "outcome": y,
                "first_treat": int(g_i) if g_i > 0 else 0,
                "eligibility": q_i,
                "treated": treated,
                "true_effect": true_eff,
            }
            if add_covariates:
                row["x1"] = x1[i]
                row["x2"] = x2[i]

            records.append(row)

    return pd.DataFrame(records)


def generate_ddd_panel_data(
    n_units: int = 200,
    n_periods: int = 8,
    treatment_period: int = 4,
    group_frac: float = 0.5,
    partition_frac: float = 0.5,
    treatment_effect: float = 2.0,
    group_effect: float = 2.0,
    partition_effect: float = 1.0,
    time_effect: float = 0.5,
    group_time_interaction: float = 1.0,
    partition_time_interaction: float = 0.5,
    group_partition_interaction: float = 1.5,
    unit_fe_sd: float = 1.5,
    noise_sd: float = 1.0,
    add_covariates: bool = False,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate synthetic panel data for Triple Difference (DDD) power analysis.

    Creates a balanced panel of n_units observed over n_periods with two
    time-invariant binary dimensions (``group`` and ``partition``) and a
    derived binary ``post`` indicator. The triple-interaction effect
    (``group * partition * post``) is the identifying ATT under DDD-CPT.

    The DGP equation is::

        Y_{i,t} = unit_fe_i
                + group_i        * group_effect
                + partition_i    * partition_effect
                + post_t         * time_effect
                + (group_i * partition_i)  * group_partition_interaction
                + (group_i * post_t)       * group_time_interaction
                + (partition_i * post_t)   * partition_time_interaction
                + treatment_effect * group_i * partition_i * post_t
                + epsilon_{i,t}

    where ``group_i`` and ``partition_i`` are unit-level (constant in t)
    and ``post_t = 1[period >= treatment_period]``. DDD-CPT identification
    holds because ``group_partition_interaction`` enters only as a
    unit-level (time-invariant) effect, leaving the triple-interaction as
    the sole source of differential group × partition trend.

    Unlike the cross-sectional ``generate_ddd_data``, this DGP provides
    panel-realistic unit fixed effects and within-unit serial structure,
    making it suitable for panel-aware power-analysis simulations or
    sanity-checking estimators that ignore the panel dimension.

    .. warning::

        ``TripleDifference`` is a repeated-cross-section ``panel=FALSE``
        estimator: its analytical default treats each row as an
        independent observation (df = n_obs - 8). When fitting against
        ``generate_ddd_panel_data`` output, the within-unit serial
        correlation makes unclustered SEs anti-conservative — they
        understate sampling variability and overstate power. Always pass
        ``cluster="unit"`` (Liang-Zeger CR1) when fitting on
        panel-generated data; the point estimate ``att`` is invariant to
        clustering but the inference contract is not. See the
        ``TripleDifference`` REGISTRY entry for the clustering contract.

    Parameters
    ----------
    n_units : int, default=200
        Number of units in the panel.
    n_periods : int, default=8
        Number of time periods.
    treatment_period : int, default=4
        Period (0-indexed) at which ``post`` switches from 0 to 1.
        Must satisfy ``1 <= treatment_period < n_periods``.
    group_frac : float, default=0.5
        Fraction of units with ``group=1``. Must be in ``(0, 1)``. The
        partition split is then drawn stratified-by-group at the requested
        ``partition_frac`` so every (group, partition) cell receives at
        least one unit; a ``ValueError`` is raised when the rounded cell
        counts would leave any cell empty.
    partition_frac : float, default=0.5
        Fraction of units with ``partition=1`` within each ``group``
        stratum. Must be in ``(0, 1)``. The stratified allocation is what
        makes TripleDifference.fit's 2x2x2 surface populated for any valid
        ``(n_units, group_frac, partition_frac)``.
    treatment_effect : float, default=2.0
        True ATT for the triple-interaction cell (group=1, partition=1,
        post=1).
    group_effect : float, default=2.0
        Main effect of ``group=1`` (unit-level).
    partition_effect : float, default=1.0
        Main effect of ``partition=1`` (unit-level).
    time_effect : float, default=0.5
        Main effect of ``post=1`` (time-level).
    group_time_interaction : float, default=1.0
        Coefficient on ``group * post`` (differential trend for the group
        dimension).
    partition_time_interaction : float, default=0.5
        Coefficient on ``partition * post`` (differential trend for the
        partition dimension).
    group_partition_interaction : float, default=1.5
        Coefficient on the unit-level ``group * partition`` interaction.
        Must be time-invariant for DDD-CPT to hold.
    unit_fe_sd : float, default=1.5
        Standard deviation of the unit fixed effect.
    noise_sd : float, default=1.0
        Standard deviation of the idiosyncratic noise term.
    add_covariates : bool, default=False
        If True, add unit-level covariates ``x1`` (continuous) and ``x2``
        (binary) that affect the outcome.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Long-format panel with columns:

        - ``unit``: integer unit ID.
        - ``period``: integer time period (0-indexed).
        - ``outcome``: outcome variable.
        - ``group``: unit-level binary group indicator (time-invariant).
        - ``partition``: unit-level binary partition indicator
          (time-invariant, orthogonal to group).
        - ``post``: binary indicator, ``1`` if ``period >= treatment_period``.
        - ``treated``: ``group * partition * post`` (binary).
        - ``true_effect``: ``treatment_effect`` when treated, else 0.
        - ``x1``, ``x2``: optional unit-level covariates (only if
          ``add_covariates=True``).

    Examples
    --------
    Generate a balanced panel with default parameters:

    >>> data = generate_ddd_panel_data(n_units=200, n_periods=8, seed=42)
    >>> data.shape
    (1600, 8)
    >>> data.groupby('unit')['period'].count().eq(8).all()
    True

    Fit with TripleDifference. Note ``time="post"`` (the derived binary
    indicator) and ``cluster="unit"`` (required for valid inference on
    panel-generated data; see the warning above):

    >>> from diff_diff import TripleDifference
    >>> result = TripleDifference(cluster="unit").fit(
    ...     data, outcome='outcome', group='group',
    ...     partition='partition', time='post',
    ... )
    """
    if not (1 <= treatment_period < n_periods):
        raise ValueError(
            f"treatment_period must satisfy 1 <= treatment_period < n_periods; "
            f"got treatment_period={treatment_period}, n_periods={n_periods}."
        )
    if not (0.0 < group_frac < 1.0):
        raise ValueError(f"group_frac must be in (0, 1); got {group_frac}.")
    if not (0.0 < partition_frac < 1.0):
        raise ValueError(f"partition_frac must be in (0, 1); got {partition_frac}.")
    if n_units < 4:
        raise ValueError(
            f"n_units must be >= 4 to populate all four (group, partition) cells; "
            f"got {n_units}."
        )

    # Stratified allocation that guarantees every (group, partition) cell receives
    # at least one unit. Independent marginal sampling could collapse a cell when
    # rounded marginals are small (e.g., n_units=4, group_frac=partition_frac=0.25
    # yields n_group_1=1 with no room for both partition values within group=1).
    # TripleDifference.fit requires all 8 G×P×T cells, so we validate first.
    n_group_1 = int(round(n_units * group_frac))
    n_group_0 = n_units - n_group_1
    n_p1_g0 = int(round(n_group_0 * partition_frac))
    n_p1_g1 = int(round(n_group_1 * partition_frac))
    n_p0_g0 = n_group_0 - n_p1_g0
    n_p0_g1 = n_group_1 - n_p1_g1
    cell_counts = {
        (0, 0): n_p0_g0,
        (0, 1): n_p1_g0,
        (1, 0): n_p0_g1,
        (1, 1): n_p1_g1,
    }
    if min(cell_counts.values()) < 1:
        raise ValueError(
            f"generate_ddd_panel_data requires every (group, partition) cell to "
            f"contain at least one unit so TripleDifference.fit's 2x2x2 surface is "
            f"populated. With n_units={n_units}, group_frac={group_frac}, "
            f"partition_frac={partition_frac}, the rounded cell counts are "
            f"(group, partition): (0,0)={n_p0_g0}, (0,1)={n_p1_g0}, "
            f"(1,0)={n_p0_g1}, (1,1)={n_p1_g1}. Increase n_units or move "
            f"group_frac/partition_frac closer to 0.5 so each cell receives "
            f">=1 unit."
        )

    rng = np.random.default_rng(seed)

    group = np.zeros(n_units, dtype=int)
    group_idx = rng.choice(n_units, size=n_group_1, replace=False)
    group[group_idx] = 1
    partition = np.zeros(n_units, dtype=int)
    g0_units = np.where(group == 0)[0]
    g1_units = np.where(group == 1)[0]
    partition[rng.choice(g0_units, size=n_p1_g0, replace=False)] = 1
    partition[rng.choice(g1_units, size=n_p1_g1, replace=False)] = 1

    unit_fe = rng.normal(0.0, unit_fe_sd, size=n_units)

    if add_covariates:
        x1 = rng.normal(0.0, 1.0, size=n_units)
        x2 = rng.choice([0, 1], size=n_units)
    else:
        x1 = None
        x2 = None

    records = []
    for i in range(n_units):
        g_i = int(group[i])
        p_i = int(partition[i])
        for t in range(n_periods):
            post_t = 1 if t >= treatment_period else 0
            y = (
                unit_fe[i]
                + g_i * group_effect
                + p_i * partition_effect
                + post_t * time_effect
                + (g_i * p_i) * group_partition_interaction
                + (g_i * post_t) * group_time_interaction
                + (p_i * post_t) * partition_time_interaction
            )
            treated = int(g_i == 1 and p_i == 1 and post_t == 1)
            true_eff = treatment_effect if treated else 0.0
            if treated:
                y += treatment_effect
            if add_covariates:
                y += 0.5 * x1[i] + 0.3 * x2[i]
            y += rng.normal(0.0, noise_sd)

            row = {
                "unit": i,
                "period": t,
                "outcome": float(y),
                "group": g_i,
                "partition": p_i,
                "post": post_t,
                "treated": treated,
                "true_effect": float(true_eff),
            }
            if add_covariates:
                row["x1"] = float(x1[i])
                row["x2"] = int(x2[i])

            records.append(row)

    return pd.DataFrame(records)


def _rank_pair_weights(
    unit_weight: np.ndarray,
    unit_stratum: np.ndarray,
    y0: np.ndarray,
    n_strata: int,
) -> None:
    """Rank-pair weights with Y(0) within each stratum (in-place).

    High-outcome units receive higher weights, modeling informative sampling
    where hard-to-reach (high-outcome) subpopulations are under-covered
    and therefore carry larger inverse-selection-probability weights.
    """
    for s in range(n_strata):
        mask = unit_stratum == s
        n_s = mask.sum()
        if n_s <= 1:
            continue
        idx_s = np.where(mask)[0]
        w_vals = unit_weight[idx_s].copy()
        if w_vals.std() < 1e-10:
            # No within-stratum variation: create rank-based weights
            # scaled to preserve stratum baseline weight level
            ranks = np.argsort(np.argsort(y0[idx_s])).astype(float) + 1.0
            unit_weight[idx_s] = ranks / ranks.mean() * w_vals.mean()
        else:
            # Rank-pair: highest Y(0) gets heaviest weight
            y0_order = np.argsort(-y0[idx_s])
            w_sorted = np.sort(w_vals)[::-1]  # heaviest first
            unit_weight[idx_s[y0_order]] = w_sorted


def generate_survey_did_data(
    n_units: int = 200,
    n_periods: int = 8,
    cohort_periods: Optional[List[int]] = None,
    never_treated_frac: float = 0.3,
    treatment_effect: float = 2.0,
    dynamic_effects: bool = False,
    effect_growth: float = 0.3,
    n_strata: int = 5,
    psu_per_stratum: int = 8,
    fpc_per_stratum: float = 200.0,
    weight_variation: str = "moderate",
    psu_re_sd: float = 2.0,
    psu_period_factor: float = 0.5,
    unit_fe_sd: float = 1.0,
    noise_sd: float = 0.5,
    include_replicate_weights: bool = False,
    add_covariates: bool = False,
    panel: bool = True,
    seed: Optional[int] = None,
    # --- Research-grade DGP parameters ---
    icc: Optional[float] = None,
    weight_cv: Optional[float] = None,
    informative_sampling: bool = False,
    heterogeneous_te_by_strata: bool = False,
    strata_sizes: Optional[List[int]] = None,
    return_true_population_att: bool = False,
    covariate_effects: Optional[tuple] = None,
    te_covariate_interaction: float = 0.0,
    conditional_pt: float = 0.0,
) -> pd.DataFrame:
    """
    Generate synthetic staggered DiD data with survey structure.

    Creates a balanced panel (or repeated cross-section) with stratified
    multi-stage sampling design (strata, PSUs, FPC, sampling weights) and
    known treatment effects. The survey structure introduces intra-cluster
    correlation via PSU random effects, making design-based SEs larger
    than naive SEs.

    Modeled on ACS/BRFSS-style stratified household surveys: strata
    represent geographic region types, PSUs are census tracts sampled
    within each stratum, and weights are inverse selection probabilities.

    Parameters
    ----------
    n_units : int, default=200
        Number of units (respondents) per period.
    n_periods : int, default=8
        Number of time periods (1-indexed).
    cohort_periods : list of int, optional
        Treatment cohort periods (1-indexed, each must be >= 2 for at least
        one pre-treatment period). Default derived from n_periods; [3, 5]
        when n_periods >= 8. Requires n_periods >= 4 when not specified.
    never_treated_frac : float, default=0.3
        Fraction of units that are never treated.
    treatment_effect : float, default=2.0
        True ATT for treated units.
    dynamic_effects : bool, default=False
        If True, effects grow over time since treatment.
    effect_growth : float, default=0.3
        Per-period effect growth rate when dynamic_effects=True.
    n_strata : int, default=5
        Number of geographic strata.
    psu_per_stratum : int, default=8
        Number of PSUs (census tracts) per stratum.
    fpc_per_stratum : float, default=200.0
        Finite population correction (total tracts per stratum).
    weight_variation : str, default="moderate"
        Controls sampling weight dispersion across strata.
        "none": all weights equal (1.0).
        "moderate": weights range ~1.0-2.0 across strata.
        "high": weights range ~1.0-4.0 across strata.
    psu_re_sd : float, default=2.0
        Standard deviation of PSU random effects. Controls intra-cluster
        correlation and drives DEFF > 1.
    psu_period_factor : float, default=0.5
        Multiplier for PSU-period interaction shocks (relative to psu_re_sd).
        Higher values increase time-varying within-cluster correlation,
        which survives DiD's time-differencing and inflates design-based SEs.
    unit_fe_sd : float, default=1.0
        Standard deviation of unit fixed effects.
    noise_sd : float, default=0.5
        Standard deviation of idiosyncratic noise.
    include_replicate_weights : bool, default=False
        If True, add JK1 (delete-one-PSU) replicate weight columns.
        Requires at least 2 PSUs.
    add_covariates : bool, default=False
        If True, add covariates x1 (continuous) and x2 (binary).
    panel : bool, default=True
        If True, generate panel data (same respondents across periods).
        If False, generate repeated cross-sections with fresh respondent
        effects and unique unit IDs each period (for use with
        CallawaySantAnna(panel=False)).
    seed : int, optional
        Random seed for reproducibility.
    icc : float, optional
        Target intra-class correlation coefficient (0 < icc < 1). Overrides
        ``psu_re_sd`` via the variance decomposition:
        ``psu_re_sd = sqrt(icc * (sigma2_unit + sigma2_noise + sigma2_cov) /
        ((1 - icc) * (1 + psu_period_factor^2)))`` where ``sigma2_cov``
        includes covariate variance when ``add_covariates=True``.
        Cannot be combined with a non-default ``psu_re_sd``.
    weight_cv : float, optional
        Target coefficient of variation for sampling weights. Generates
        LogNormal weights normalized to mean 1, bypassing ``weight_variation``.
        Cannot be combined with a non-default ``weight_variation``.
    informative_sampling : bool, default=False
        If True, sampling weights correlate with Y(0) — high-outcome units
        receive higher weights (under-coverage → larger inverse-selection-
        probability weights). Uses rank-pairing within each stratum. For
        panel data, ranking is done once from period-1 outcomes. For
        repeated cross-sections, ranking is refreshed each period. Within
        each stratum, rank-based weights are scaled to preserve the
        stratum's baseline weight level from ``weight_variation``.
        When ``add_covariates=True``, covariate contributions are
        included in the Y(0) ranking.
    heterogeneous_te_by_strata : bool, default=False
        If True, treatment effect varies by stratum:
        ``TE_h = TE * (1 + 0.5 * (h - mean) / std)``. Creates a gap
        between unweighted and population ATT. With ``n_strata=1``,
        all units receive the base ``treatment_effect``.
    strata_sizes : list of int, optional
        Custom per-stratum unit counts. Must have length ``n_strata`` and
        sum to ``n_units``. Replaces equal allocation across strata.
    return_true_population_att : bool, default=False
        If True, attaches a diagnostic dict to ``df.attrs["dgp_truth"]``
        with keys: ``population_att`` (weight-weighted average of treated
        true effects), ``deff_kish`` (1 + CV(w)^2), ``base_stratum_effects``
        (base stratum TEs before dynamic/covariate modifiers),
        ``icc_realized`` (ANOVA-based ICC computed on period-1 data),
        and ``conditional_pt_active`` (bool, whether conditional PT
        regime is active).
    covariate_effects : tuple of (float, float), optional
        Coefficients ``(beta1, beta2)`` for covariates x1 and x2 in the
        outcome equation ``y += beta1 * x1 + beta2 * x2``. Default uses
        ``(0.5, 0.3)``. Only used when ``add_covariates=True``. The ICC
        calibration automatically adjusts for the implied covariate variance.
    te_covariate_interaction : float, default=0.0
        Coefficient for treatment-by-covariate interaction:
        ``TE_i = base_TE + te_covariate_interaction * x1_i``. Creates
        unit-level treatment effect heterogeneity driven by the continuous
        covariate. Requires ``add_covariates=True``.
    conditional_pt : float, default=0.0
        Coefficient for X-dependent time trend:
        ``y += conditional_pt * x1_i * (t / n_periods)``. When nonzero,
        treated units' x1 is drawn from N(1, 1) instead of N(0, 1),
        creating differential pre-trends correlated with covariates.
        Conditional on x1, trends remain parallel (conditional PT holds).
        DR/IPW estimators with covariates recover truth; no-covariate
        estimators are biased. Uses normalized time (t/n_periods) for
        scale independence. Requires ``add_covariates=True`` and at least
        one ever-treated and one never-treated unit (the x1 mean shift
        only differentiates ever-treated from never-treated units).

        .. note:: When used with ``icc``, the ICC calibration is approximate
           because the x1 mean shift creates a mixture distribution with
           slightly higher marginal variance than the assumed Var(x1) = 1.

    Returns
    -------
    pd.DataFrame
        Columns: unit, period, outcome, first_treat, treated, true_effect,
        stratum, psu, fpc, weight. Also rep_0..rep_K if
        include_replicate_weights=True, and x1, x2 if add_covariates=True.
        If ``return_true_population_att=True``, ``df.attrs["dgp_truth"]``
        contains DGP diagnostics.
    """
    rng = np.random.default_rng(seed)

    # --- Upfront parameter validation ---
    if n_units < 1:
        raise ValueError(f"n_units must be positive, got {n_units}")
    if n_periods < 1:
        raise ValueError(f"n_periods must be positive, got {n_periods}")
    if n_strata < 1:
        raise ValueError(f"n_strata must be positive, got {n_strata}")
    if psu_per_stratum < 1:
        raise ValueError(f"psu_per_stratum must be positive, got {psu_per_stratum}")
    if not 0.0 <= never_treated_frac <= 1.0:
        raise ValueError(f"never_treated_frac must be between 0 and 1, got {never_treated_frac}")
    if fpc_per_stratum < psu_per_stratum:
        raise ValueError(
            f"fpc_per_stratum ({fpc_per_stratum}) must be >= psu_per_stratum "
            f"({psu_per_stratum})"
        )

    if cohort_periods is None:
        # Derive defaults from n_periods.  Cohorts need g >= 2 (at least one
        # pre-period for estimability with CallawaySantAnna).
        if n_periods >= 8:
            cohort_periods = [3, 5]
        elif n_periods >= 4:
            cohort_periods = [max(2, n_periods // 3), max(3, 2 * n_periods // 3)]
        else:
            raise ValueError(
                f"n_periods={n_periods} is too small for default cohort_periods "
                f"(need n_periods >= 4 for at least one cohort with a pre-period). "
                f"Pass cohort_periods explicitly for small panels."
            )
    # Coerce array-like to list (handles np.array inputs)
    cohort_periods = list(cohort_periods)
    if not cohort_periods:
        raise ValueError("cohort_periods must be a non-empty list of integers")
    for cp in cohort_periods:
        if isinstance(cp, bool) or not isinstance(cp, (int, np.integer)):
            raise ValueError(f"cohort_periods must contain integers, got {cp!r}")
        if cp < 2 or cp > n_periods:
            raise ValueError(
                f"Cohort period {cp} must be between 2 and {n_periods} "
                f"(g >= 2 ensures at least one pre-treatment period)"
            )

    if not np.isfinite(psu_period_factor) or psu_period_factor < 0:
        raise ValueError(
            f"psu_period_factor must be finite and non-negative, " f"got {psu_period_factor}"
        )

    valid_wv = ("none", "moderate", "high")
    if weight_variation not in valid_wv:
        raise ValueError(f"weight_variation must be one of {valid_wv}, got {weight_variation!r}")

    # --- Validate research-grade DGP parameters ---
    if icc is not None:
        if not (0 < icc < 1):
            raise ValueError(f"icc must be between 0 and 1 (exclusive), got {icc}")
        if psu_re_sd != 2.0:
            raise ValueError(
                "Cannot specify both icc and a non-default psu_re_sd. "
                "icc overrides psu_re_sd via the ICC formula."
            )

    if weight_cv is not None:
        if not np.isfinite(weight_cv) or weight_cv <= 0:
            raise ValueError(f"weight_cv must be finite and positive, got {weight_cv}")
        if weight_variation != "moderate":
            raise ValueError(
                "Cannot specify both weight_cv and a non-default "
                "weight_variation. weight_cv overrides weight_variation."
            )

    if strata_sizes is not None:
        strata_sizes = list(strata_sizes)
        for ss in strata_sizes:
            if isinstance(ss, bool) or not isinstance(ss, (int, np.integer)):
                raise ValueError(f"strata_sizes must contain integers, got {ss!r}")
        if len(strata_sizes) != n_strata:
            raise ValueError(
                f"strata_sizes must have length n_strata={n_strata}, " f"got {len(strata_sizes)}"
            )
        if any(s < 1 for s in strata_sizes):
            raise ValueError("All strata_sizes must be >= 1")
        if sum(strata_sizes) != n_units:
            raise ValueError(
                f"strata_sizes must sum to n_units={n_units}, " f"got {sum(strata_sizes)}"
            )

    # --- Validate and resolve covariate coefficients ---
    if covariate_effects is not None:
        covariate_effects = tuple(covariate_effects)
        if len(covariate_effects) != 2:
            raise ValueError(f"covariate_effects must have length 2, got {len(covariate_effects)}")
        if not all(np.isfinite(c) for c in covariate_effects):
            raise ValueError(f"covariate_effects must be finite, got {covariate_effects}")
    _beta1, _beta2 = covariate_effects if covariate_effects is not None else (0.5, 0.3)

    if not np.isfinite(te_covariate_interaction):
        raise ValueError(f"te_covariate_interaction must be finite, got {te_covariate_interaction}")
    if te_covariate_interaction != 0.0 and not add_covariates:
        raise ValueError("te_covariate_interaction requires add_covariates=True")

    if not np.isfinite(conditional_pt):
        raise ValueError(f"conditional_pt must be finite, got {conditional_pt}")
    if conditional_pt != 0.0 and not add_covariates:
        raise ValueError("conditional_pt requires add_covariates=True")
    if conditional_pt != 0.0:
        n_never = int(n_units * never_treated_frac)
        n_treated = n_units - n_never
        if n_never < 1 or n_treated < 1:
            raise ValueError(
                "conditional_pt requires at least one ever-treated and one "
                f"never-treated unit (n_units={n_units}, "
                f"never_treated_frac={never_treated_frac} yields "
                f"{n_never} never-treated, {n_treated} treated). "
                "The x1 mean shift differentiates ever-treated from "
                "never-treated units; both groups must be present."
            )

    # --- ICC -> psu_re_sd resolution ---
    if icc is not None:
        # Covariate variance: Var(beta1*x1) + Var(beta2*x2)
        # where x1 ~ N(0,1), x2 ~ Bernoulli(0.5)
        cov_var = (_beta1**2 * 1.0 + _beta2**2 * 0.25) if add_covariates else 0.0
        non_psu_var = unit_fe_sd**2 + noise_sd**2 + cov_var
        if non_psu_var < 1e-12:
            raise ValueError(
                "icc requires non-zero non-PSU variance "
                "(unit_fe_sd, noise_sd, or add_covariates must contribute variance)"
            )
        psu_re_sd = np.sqrt(icc * non_psu_var / ((1 - icc) * (1 + psu_period_factor**2)))

    # --- Survey structure: assign units to strata and PSUs ---
    n_psu_total = n_strata * psu_per_stratum

    if strata_sizes is not None:
        stratum_n = strata_sizes
    else:
        units_per_stratum = n_units // n_strata
        remainder = n_units % n_strata
        stratum_n = [units_per_stratum + (1 if s < remainder else 0) for s in range(n_strata)]

    unit_stratum = np.empty(n_units, dtype=int)
    unit_psu = np.empty(n_units, dtype=int)
    idx = 0
    for s in range(n_strata):
        n_s = stratum_n[s]
        unit_stratum[idx : idx + n_s] = s
        psu_start = s * psu_per_stratum
        for j in range(n_s):
            unit_psu[idx + j] = psu_start + (j % psu_per_stratum)
        idx += n_s

    # Sampling weights
    if weight_cv is not None:
        sigma_ln = np.sqrt(np.log(1 + weight_cv**2))
        raw_w = rng.lognormal(-(sigma_ln**2) / 2, sigma_ln, size=n_units)
        unit_weight = raw_w / raw_w.mean()
    else:
        # Stratum-based weights (inverse selection probability)
        scale_map = {"none": 0.0, "moderate": 1.0, "high": 3.0}
        scale = scale_map.get(weight_variation, 1.0)
        denom = max(n_strata - 1, 1)
        unit_weight = 1.0 + scale * (unit_stratum / denom)

    # --- Treatment assignment (cohort structure) ---
    n_never = int(n_units * never_treated_frac)
    n_treated_total = n_units - n_never
    n_per_cohort = n_treated_total // len(cohort_periods)

    unit_cohort = np.zeros(n_units, dtype=int)
    ci = n_never
    for i, g in enumerate(cohort_periods):
        n_g = n_per_cohort if i < len(cohort_periods) - 1 else n_treated_total - ci + n_never
        unit_cohort[ci : ci + n_g] = g
        ci += n_g

    # --- JK1 early guard (configured count; populated count checked after build) ---
    if include_replicate_weights and n_psu_total < 2:
        raise ValueError("JK1 replicate weights require at least 2 PSUs, " f"got {n_psu_total}.")

    # --- Random effects ---
    psu_re = rng.normal(0, psu_re_sd, size=n_psu_total)
    # PSU-period shocks: intra-cluster correlation that survives first-
    # differencing in DiD.  Without these, the time-invariant PSU RE
    # cancels in the treatment-vs-control time-difference and the
    # cluster-robust / survey SE would be *smaller* than naive OLS SE.
    # Controlled by psu_period_factor (default 0.5); higher values
    # increase time-varying clustering and inflate design-based SEs.
    psu_period_re = rng.normal(0, psu_re_sd * psu_period_factor, size=(n_psu_total, n_periods))

    # --- Informative sampling (panel path): pre-draw FEs, rank-pair weights ---
    if informative_sampling and panel:
        _panel_unit_fe = rng.normal(0, unit_fe_sd, size=n_units)
        y0_period1 = _panel_unit_fe + psu_re[unit_psu] + psu_period_re[unit_psu, 0] + 0.5
        if add_covariates:
            _panel_x1 = rng.normal(0, 1, size=n_units)
            if conditional_pt != 0.0:
                _panel_x1[unit_cohort > 0] += 1.0
            _panel_x2 = rng.choice([0, 1], size=n_units)
            y0_period1 = y0_period1 + _beta1 * _panel_x1 + _beta2 * _panel_x2
            if conditional_pt != 0.0:
                y0_period1 = y0_period1 + conditional_pt * _panel_x1 * (1 / n_periods)
        _rank_pair_weights(unit_weight, unit_stratum, y0_period1, n_strata)

    # Save base weights for cross-section informative sampling (reset each period)
    if informative_sampling and not panel:
        _base_weight = unit_weight.copy()

    # --- Heterogeneous treatment effects by stratum ---
    if heterogeneous_te_by_strata:
        if n_strata == 1:
            te_by_stratum = np.array([treatment_effect])
        else:
            strata_idx = np.arange(n_strata, dtype=float)
            te_by_stratum = treatment_effect * (
                1 + 0.5 * (strata_idx - strata_idx.mean()) / strata_idx.std()
            )
    else:
        te_by_stratum = None

    # --- Generate panel or repeated cross-sections ---
    records = []
    for t in range(1, n_periods + 1):
        # For repeated cross-sections, draw fresh respondent effects each period
        unit_fe = rng.normal(0, unit_fe_sd, size=n_units)
        if panel and t > 1:
            pass  # reuse unit_fe from first period (set below)
        if informative_sampling and panel:
            unit_fe = _panel_unit_fe  # use pre-drawn FEs
        elif panel and t == 1:
            _panel_unit_fe = unit_fe  # save for reuse
        elif panel and t > 1:
            unit_fe = _panel_unit_fe  # type: ignore[possibly-undefined]

        # Cross-section informative sampling: re-rank weights each period
        if informative_sampling and not panel:
            # Draw covariates early so they can be included in Y(0) ranking
            if add_covariates:
                x1 = rng.normal(0, 1, size=n_units)
                if conditional_pt != 0.0:
                    x1[unit_cohort > 0] += 1.0
                x2 = rng.choice([0, 1], size=n_units)
            unit_weight = _base_weight.copy()  # type: ignore[possibly-undefined]
            y0_t = unit_fe + psu_re[unit_psu] + psu_period_re[unit_psu, t - 1] + 0.5 * t
            if add_covariates:
                y0_t = y0_t + _beta1 * x1 + _beta2 * x2
                if conditional_pt != 0.0:
                    y0_t = y0_t + conditional_pt * x1 * (t / n_periods)
            _rank_pair_weights(unit_weight, unit_stratum, y0_t, n_strata)

        # Covariates — may already be drawn by informative sampling above
        if informative_sampling and panel and add_covariates:
            x1 = _panel_x1  # pre-drawn before loop for ranking
            x2 = _panel_x2
        elif informative_sampling and not panel and add_covariates:
            pass  # x1, x2 already drawn in cross-section ranking block
        elif add_covariates:
            x1 = rng.normal(0, 1, size=n_units)
            if conditional_pt != 0.0:
                x1[unit_cohort > 0] += 1.0
            x2 = rng.choice([0, 1], size=n_units)
        else:
            x1 = None
            x2 = None
        if not informative_sampling and panel and t > 1 and add_covariates:
            x1 = _panel_x1  # type: ignore[possibly-undefined]
            x2 = _panel_x2  # type: ignore[possibly-undefined]
        elif not informative_sampling and panel and t == 1 and add_covariates:
            _panel_x1 = x1
            _panel_x2 = x2

        for i in range(n_units):
            g_i = unit_cohort[i]
            # Outcome: unit FE + PSU RE + PSU-period shock + time trend
            y = unit_fe[i] + psu_re[unit_psu[i]] + psu_period_re[unit_psu[i], t - 1] + 0.5 * t

            if add_covariates:
                y += _beta1 * x1[i] + _beta2 * x2[i]
                if conditional_pt != 0.0:
                    y += conditional_pt * x1[i] * (t / n_periods)

            treated = int(g_i > 0 and t >= g_i)
            true_eff = 0.0
            if treated:
                if te_by_stratum is not None:
                    true_eff = float(te_by_stratum[unit_stratum[i]])
                else:
                    true_eff = treatment_effect
                if te_covariate_interaction != 0.0:
                    true_eff += te_covariate_interaction * x1[i]
                if dynamic_effects:
                    true_eff *= 1 + effect_growth * (t - g_i)
                y += true_eff

            y += rng.normal(0, noise_sd)

            # In cross-section mode, each period gets unique unit IDs
            uid = i if panel else (t - 1) * n_units + i

            row = {
                "unit": uid,
                "period": t,
                "outcome": y,
                "first_treat": g_i,
                "treated": treated,
                "true_effect": true_eff,
                "stratum": int(unit_stratum[i]),
                "psu": int(unit_psu[i]),
                "fpc": fpc_per_stratum,
                "weight": float(unit_weight[i]),
            }
            if add_covariates:
                row["x1"] = x1[i]
                row["x2"] = x2[i]
            records.append(row)

    df = pd.DataFrame(records)

    # --- Replicate weights (JK1 delete-one-PSU) ---
    if include_replicate_weights:
        psu_ids = sorted(df["psu"].unique())
        n_rep = len(psu_ids)
        if n_rep < 2:
            raise ValueError(
                "JK1 replicate weights require at least 2 populated PSUs, "
                f"got {n_rep}. Increase n_units or decrease psu_per_stratum."
            )
        base_w = df["weight"].values
        for r, psu_id in enumerate(psu_ids):
            w_r = base_w.copy()
            mask = df["psu"].values == psu_id
            w_r[mask] = 0.0
            # Rescale remaining: k/(k-1) for JK1
            w_r[w_r > 0] *= n_rep / (n_rep - 1)
            df[f"rep_{r}"] = w_r

    # --- DGP truth diagnostics ---
    if return_true_population_att:
        treated_mask = df["treated"] == 1
        if treated_mask.any():
            w_treated = df.loc[treated_mask, "weight"].values
            te_treated = df.loc[treated_mask, "true_effect"].values
            population_att = float(np.average(te_treated, weights=w_treated))
        else:
            population_att = float("nan")

        if te_by_stratum is not None:
            stratum_effects = {int(s): float(te_by_stratum[s]) for s in range(n_strata)}
        else:
            stratum_effects = {int(s): float(treatment_effect) for s in range(n_strata)}

        # Kish DEFF from weight variation
        w_all = df.groupby("unit")["weight"].first().values
        cv_w = float(w_all.std() / w_all.mean()) if w_all.mean() > 0 else 0.0
        deff_kish = 1 + cv_w**2

        # Realized ICC (ANOVA-based, period-1 only to avoid TE contamination)
        _p1 = df[df["period"] == 1]
        _groups = _p1.groupby("psu")["outcome"]
        _n_total = len(_p1)
        _n_groups = _groups.ngroups
        # ICC undefined with < 2 groups or no within-group replication
        if _n_groups < 2 or _n_total <= _n_groups:
            icc_realized = float("nan")
        else:
            _n_bar = _n_total / _n_groups
            _grand_mean = _p1["outcome"].mean()
            _ssb = (_groups.size() * (_groups.mean() - _grand_mean) ** 2).sum()
            _msb = _ssb / (_n_groups - 1)
            _ssw = _groups.apply(lambda x: ((x - x.mean()) ** 2).sum()).sum()
            _msw = _ssw / (_n_total - _n_groups)
            _denom = _msb + (_n_bar - 1) * _msw
            icc_realized = float((_msb - _msw) / _denom) if _denom > 0 else float("nan")

        df.attrs["dgp_truth"] = {
            "population_att": population_att,
            "deff_kish": float(deff_kish),
            "base_stratum_effects": stratum_effects,
            "icc_realized": icc_realized,
            "conditional_pt_active": conditional_pt != 0.0,
        }

    return df


# =============================================================================
# Reversible-treatment data generator (dCDH / ChaisemartinDHaultfoeuille)
# =============================================================================


def _generate_reversible_treatment_matrix(
    pattern: str,
    n_groups: int,
    n_periods: int,
    p_switch: float,
    initial_treat_frac: float,
    cycle_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Internal helper for ``generate_reversible_did_data``.

    Returns an ``(n_groups, n_periods)`` int array of binary treatment values.
    """
    D = np.zeros((n_groups, n_periods), dtype=int)

    if pattern == "single_switch":
        # Mix of joiners and leavers based on initial_treat_frac.
        # Each group switches exactly once at a uniform-random time in [1, n_periods - 1].
        initial_treated = rng.random(n_groups) < initial_treat_frac
        switch_times = rng.integers(1, n_periods, size=n_groups)
        for g in range(n_groups):
            if initial_treated[g]:
                # Starts treated, switches to untreated at switch_times[g]
                D[g, : switch_times[g]] = 1
                D[g, switch_times[g] :] = 0
            else:
                # Starts untreated, switches to treated at switch_times[g]
                D[g, : switch_times[g]] = 0
                D[g, switch_times[g] :] = 1

    elif pattern == "joiners_only":
        # All groups start untreated, each switches to treated once at random time
        switch_times = rng.integers(1, n_periods, size=n_groups)
        for g in range(n_groups):
            D[g, switch_times[g] :] = 1

    elif pattern == "leavers_only":
        # All groups start treated, each switches to untreated once at random time
        switch_times = rng.integers(1, n_periods, size=n_groups)
        for g in range(n_groups):
            D[g, : switch_times[g]] = 1

    elif pattern == "mixed_single_switch":
        # Deterministic: first half are joiners, second half are leavers.
        # Each group switches exactly once at a uniform-random time.
        switch_times = rng.integers(1, n_periods, size=n_groups)
        n_joiners = n_groups // 2
        for g in range(n_groups):
            if g < n_joiners:
                D[g, switch_times[g] :] = 1  # Joiner
            else:
                D[g, : switch_times[g]] = 1  # Leaver

    elif pattern == "random":
        # Initial state random, then flip with probability p_switch each subsequent period.
        # Often produces multi-switch groups for n_periods >= 4 and p_switch > 0.
        D[:, 0] = (rng.random(n_groups) < initial_treat_frac).astype(int)
        for t in range(1, n_periods):
            flips = rng.random(n_groups) < p_switch
            D[:, t] = np.where(flips, 1 - D[:, t - 1], D[:, t - 1])

    elif pattern == "cycles":
        # Deterministic on/off cycles of length cycle_length.
        # Half the groups start in the "0" phase, half start in the "1" phase.
        # All groups are multi-switch when n_periods > 2 * cycle_length.
        for t in range(n_periods):
            phase = (t // cycle_length) % 2
            n_first_half = n_groups // 2
            D[:n_first_half, t] = phase
            D[n_first_half:, t] = 1 - phase

    elif pattern == "marketing":
        # Seasonal "2 on, 1 off" pattern, identical for all groups.
        # All groups are multi-switch when n_periods >= 4.
        for t in range(n_periods):
            phase_in_cycle = t % 3
            on = phase_in_cycle != 2
            D[:, t] = int(on)

    return D


def generate_reversible_did_data(
    n_groups: int = 50,
    n_periods: int = 6,
    pattern: str = "single_switch",
    p_switch: float = 0.2,
    initial_treat_frac: float = 0.3,
    cycle_length: int = 2,
    treatment_effect: float = 2.0,
    heterogeneous_effects: bool = False,
    effect_sd: float = 0.5,
    group_fe_sd: float = 2.0,
    time_trend: float = 0.1,
    noise_sd: float = 0.5,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate synthetic panel data with reversible (non-absorbing) treatment.

    Treatment can switch on and off over time, supporting designs where the
    canonical staggered-adoption assumption (once treated, always treated)
    does not hold. This is the only generator in the library that produces
    reversible-treatment data; intended for the
    :class:`~diff_diff.ChaisemartinDHaultfoeuille` (dCDH) estimator.

    Seven patterns are supported. Four of them are guaranteed to keep every
    group as a "single-switch" group (each group switches treatment status
    at most once), so the dCDH ``drop_larger_lower=True`` filter is a no-op.
    The other three deliberately produce multi-switch groups for stress-
    testing the drop logic.

    Parameters
    ----------
    n_groups : int, default=50
        Number of groups in the panel.
    n_periods : int, default=6
        Number of time periods. Must be at least 2.
    pattern : str, default="single_switch"
        Treatment pattern. One of:

        - ``"single_switch"`` (default, single-switch): each group switches
          exactly once at a uniform-random time. Mix of joiners and leavers
          determined by ``initial_treat_frac``.
        - ``"joiners_only"`` (single-switch): all groups start untreated and
          each switches to treated once. Pure staggered adoption.
        - ``"leavers_only"`` (single-switch): mirror of ``joiners_only`` —
          all groups start treated and each switches to untreated once.
        - ``"mixed_single_switch"`` (single-switch): deterministic 50/50 mix
          of joiners and leavers, each with exactly one switch. Useful for
          parity tests where you want a guaranteed split independent of seed.
        - ``"random"`` (often multi-switch): each ``(g, t >= 1)`` flips
          treatment from the previous period with probability ``p_switch``.
          Initial state drawn from ``Bernoulli(initial_treat_frac)``. With
          ``n_periods >= 4`` and ``p_switch > 0``, many groups will switch
          more than once and will be dropped under
          ``drop_larger_lower=True``. Useful for stress-testing the drop
          filter.
        - ``"cycles"`` (always multi-switch): deterministic on/off cycles of
          length ``cycle_length``. Half the groups start in the "0" phase
          and half in the "1" phase, so the panel always contains both
          joiner and leaver transitions. Every group is multi-switch when
          ``n_periods > 2 * cycle_length``.
        - ``"marketing"`` (always multi-switch): seasonal "2 on, 1 off"
          pattern starting in the on phase, identical across groups. Mimics
          a marketing campaign with periodic breaks.
    p_switch : float, default=0.2
        Per-period flip probability. Only used when ``pattern="random"``.
    initial_treat_frac : float, default=0.3
        Fraction of groups starting in the treated state at period 0. Only
        used by ``"single_switch"`` and ``"random"``.
    cycle_length : int, default=2
        Length of each on/off phase. Only used when ``pattern="cycles"``.
    treatment_effect : float, default=2.0
        Average treatment effect on treated cells. With
        ``heterogeneous_effects=False``, every treated cell has exactly this
        effect. With ``True``, this is the mean of a Normal distribution.
    heterogeneous_effects : bool, default=False
        If True, per-cell true effects are drawn independently from
        ``Normal(treatment_effect, effect_sd)``.
    effect_sd : float, default=0.5
        Standard deviation of per-cell effects when
        ``heterogeneous_effects=True``.
    group_fe_sd : float, default=2.0
        Standard deviation of group fixed effects.
    time_trend : float, default=0.1
        Linear time trend coefficient.
    noise_sd : float, default=0.5
        Standard deviation of idiosyncratic noise.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Synthetic balanced panel with one row per ``(group, period)`` cell
        and the following columns:

        - ``group`` (int): group identifier in ``[0, n_groups)``
        - ``period`` (int): time period in ``[0, n_periods)``
        - ``treatment`` (int): per-cell binary treatment (0 or 1)
        - ``outcome`` (float): outcome variable
        - ``true_effect`` (float): per-cell true treatment effect (0 if
          untreated)
        - ``d_lag`` (float): previous-period treatment, NaN at period 0
        - ``switcher_type`` (object): one of ``"initial"`` (period 0),
          ``"joiner"`` (``d_lag=0, treatment=1``), ``"leaver"``
          (``d_lag=1, treatment=0``), ``"stable_0"``
          (``d_lag=0, treatment=0``), or ``"stable_1"``
          (``d_lag=1, treatment=1``)

    Notes
    -----
    The default pattern is ``"single_switch"`` so the generator's happy path
    produces data that the dCDH estimator can use directly without dropping
    groups. The ``"random"``, ``"cycles"``, and ``"marketing"`` patterns are
    primarily for stress-testing the ``drop_larger_lower=True`` filter and
    will produce data where many or all groups are filtered out before
    estimation.

    The default ``pattern="single_switch"`` is **A5-safe by construction**:
    every group has at most one transition, so no group can be a "crosser"
    that switches in and back out. The dCDH estimator's
    ``drop_larger_lower=True`` filter (matching R ``DIDmultiplegtDYN``) is
    a no-op on this pattern. Other patterns (``random``, ``cycles``,
    ``marketing``) ARE allowed to violate A5 and are useful primarily for
    stress-testing the multi-switch drop filter — passing them through the
    estimator with ``drop_larger_lower=True`` should drop a non-zero count
    of crosser groups, which is the intended check. The cohort-recentered
    variance formula in Web Appendix Section 3.7.3 of the dynamic
    companion paper is derived under A5, which is why the drop filter is
    on by default.

    Examples
    --------
    Default single-switch panel (mix of joiners and leavers, all groups
    survive ``drop_larger_lower=True``):

    >>> data = generate_reversible_did_data(n_groups=20, n_periods=6, seed=42)
    >>> sorted(data.columns.tolist())
    ['d_lag', 'group', 'outcome', 'period', 'switcher_type', 'treatment', 'true_effect']
    >>> set(data['switcher_type']).issubset(
    ...     {'initial', 'joiner', 'leaver', 'stable_0', 'stable_1'}
    ... )
    True

    Joiners-only (pure staggered adoption):

    >>> data = generate_reversible_did_data(
    ...     n_groups=20, pattern="joiners_only", seed=1
    ... )
    >>> set(data.query("period == 0")['treatment'].unique()) == {0}
    True

    Leavers-only:

    >>> data = generate_reversible_did_data(
    ...     n_groups=20, pattern="leavers_only", seed=2
    ... )
    >>> set(data.query("period == 0")['treatment'].unique()) == {1}
    True
    """
    # --- Parameter validation ---
    valid_patterns = {
        "single_switch",
        "joiners_only",
        "leavers_only",
        "mixed_single_switch",
        "random",
        "cycles",
        "marketing",
    }
    if pattern not in valid_patterns:
        raise ValueError(f"pattern must be one of {sorted(valid_patterns)}, got {pattern!r}")
    if n_groups < 1:
        raise ValueError(f"n_groups must be positive, got {n_groups}")
    if n_periods < 2:
        raise ValueError(f"n_periods must be at least 2, got {n_periods}")
    if not 0.0 <= initial_treat_frac <= 1.0:
        raise ValueError(f"initial_treat_frac must be in [0, 1], got {initial_treat_frac}")
    if not 0.0 <= p_switch <= 1.0:
        raise ValueError(f"p_switch must be in [0, 1], got {p_switch}")
    if cycle_length < 1:
        raise ValueError(f"cycle_length must be positive, got {cycle_length}")
    if effect_sd < 0:
        raise ValueError(f"effect_sd must be non-negative, got {effect_sd}")
    if group_fe_sd < 0:
        raise ValueError(f"group_fe_sd must be non-negative, got {group_fe_sd}")
    if noise_sd < 0:
        raise ValueError(f"noise_sd must be non-negative, got {noise_sd}")

    rng = np.random.default_rng(seed)

    # --- Generate the (n_groups, n_periods) treatment matrix ---
    D = _generate_reversible_treatment_matrix(
        pattern=pattern,
        n_groups=n_groups,
        n_periods=n_periods,
        p_switch=p_switch,
        initial_treat_frac=initial_treat_frac,
        cycle_length=cycle_length,
        rng=rng,
    )

    # --- Generate fixed effects, true effects, outcomes ---
    group_fe = rng.normal(0, group_fe_sd, n_groups)
    if heterogeneous_effects:
        true_effects = rng.normal(treatment_effect, effect_sd, (n_groups, n_periods))
    else:
        true_effects = np.full((n_groups, n_periods), float(treatment_effect))
    # Only treated cells have non-zero effect
    true_effects = np.where(D == 1, true_effects, 0.0)

    period_arr = np.arange(n_periods)
    Y = (
        10.0
        + group_fe[:, None]
        + time_trend * period_arr[None, :]
        + true_effects
        + rng.normal(0, noise_sd, (n_groups, n_periods))
    )

    # --- Compute d_lag (NaN at period 0) ---
    D_lag = np.full((n_groups, n_periods), np.nan)
    D_lag[:, 1:] = D[:, :-1]

    # --- Vectorized switcher_type classification ---
    treatment_flat = D.flatten()
    d_lag_flat = D_lag.flatten()
    switcher_type = np.full(n_groups * n_periods, "stable_1", dtype=object)
    # Order matters: more specific masks last so they overwrite the default
    mask_stable_0 = (d_lag_flat == 0) & (treatment_flat == 0)
    mask_joiner = (d_lag_flat == 0) & (treatment_flat == 1)
    mask_leaver = (d_lag_flat == 1) & (treatment_flat == 0)
    mask_initial = np.isnan(d_lag_flat)
    switcher_type[mask_stable_0] = "stable_0"
    switcher_type[mask_joiner] = "joiner"
    switcher_type[mask_leaver] = "leaver"
    switcher_type[mask_initial] = "initial"  # always wins for period 0

    # --- Build the long-format DataFrame ---
    return pd.DataFrame(
        {
            "group": np.repeat(np.arange(n_groups), n_periods),
            "period": np.tile(period_arr, n_groups),
            "treatment": treatment_flat,
            "outcome": Y.flatten(),
            "true_effect": true_effects.flatten(),
            "d_lag": d_lag_flat,
            "switcher_type": switcher_type,
        }
    )
