"""
Data generation utilities for difference-in-differences analysis.

This module provides functions to generate synthetic datasets for testing
and validating DiD estimators, including basic 2x2 DiD, staggered adoption,
factor model data, triple difference, and event study designs.
"""

from typing import List, Optional

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
    seed: Optional[int] = None
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

            records.append({
                "unit": unit,
                "period": period,
                "treated": int(is_treated),
                "post": int(is_post),
                "outcome": y,
                "true_effect": effect
            })

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
            raise ValueError(
                f"Cohort period {cp} must be between 1 and {n_periods - 1}"
            )

    # Determine number of never-treated and treated units
    n_never = int(n_units * never_treated_frac)
    n_treated = n_units - n_never

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

            records.append({
                "unit": unit,
                "period": period,
                "outcome": y,
                "first_treat": unit_first_treat,
                "treated": int(is_treated),
                "treat": int(is_ever_treated),
                "true_effect": effect,
            })

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

    n_control = n_units - n_treated
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

            records.append({
                "unit": i,
                "period": t,
                "outcome": y,
                "treated": int(is_ever_treated and post),
                "treat": int(is_ever_treated),
                "true_effect": effect,
            })

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
                    y += 1.5 * g * p   # group-partition interaction
                    y += 1.0 * g * t   # group-time interaction (diff trends)
                    y += 0.5 * p * t   # partition-time interaction

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

            records.append({
                "unit": unit,
                "period": period,
                "treated": int(is_treated),
                "post": int(post),
                "outcome": y,
                "true_effect": effect,
            })

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

            records.append({
                "unit": unit,
                "period": period,
                "treated": int(is_treated),
                "post": int(post),
                "outcome": y,
                "event_time": event_time,
                "true_effect": effect,
            })

    return pd.DataFrame(records)
