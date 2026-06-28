"""Synthetic data-generating processes for spillover-DiD tests.

These factories produce panels satisfying Butts (2021) Assumptions 1, 3, 5,
and 7 by construction. Used by ``tests/test_spillover.py`` to anchor
identification claims (Wave B MVP correctness anchors).
"""

from typing import Callable, Optional

import numpy as np
import pandas as pd


def generate_butts_nonstaggered_dgp(
    *,
    n_units: int = 200,
    n_treated: int = 20,
    n_near_control: int = 80,
    n_far_control: int = 100,
    n_periods: int = 2,
    t_treat: int = 1,
    tau_total: float = -0.07,
    delta_1: float = -0.04,
    d_bar: float = 100.0,
    near_control_d_max: Optional[float] = None,
    far_control_d_min: Optional[float] = None,
    parallel_trends_shock_sd: float = 0.0,
    error_sd: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """Non-staggered panel with known direct + spillover effects.

    All treated units share onset at ``t_treat``. Layout:

    - Treated units cluster near (lat=0, lon=0).
    - Near-controls placed within ``near_control_d_max`` (default
      ``d_bar / 2`` km) so they're inside the spillover zone.
    - Far-controls placed beyond ``far_control_d_min`` (default
      ``2 * d_bar`` km).

    Potential-outcomes model (satisfies Assumption 6/7 by construction):

    .. code::

        Y_it(0, 0) = mu_i + lambda_t + e_it
        Y_i1(1, 0) - Y_i0(0, 0) = (lambda_1 - lambda_0) + tau_total
                                    for treated units
        Y_i1(0, S_i=1) - Y_i0(0, S_i=1) = (lambda_1 - lambda_0) + delta_1
                                    for near-control units (D_i=0, S_i=1)
        Y_i1(0, S_i=0) - Y_i0(0, S_i=0) = (lambda_1 - lambda_0)
                                    for far-control units (clean)

    Returns
    -------
    DataFrame with columns: ``unit``, ``time``, ``D`` (binary at row level),
    ``first_treat`` (per-unit onset; ``np.inf`` for never-treated),
    ``y`` (outcome), ``lat``, ``lon``.
    """
    if n_units != n_treated + n_near_control + n_far_control:
        n_units = n_treated + n_near_control + n_far_control

    if near_control_d_max is None:
        near_control_d_max = d_bar / 2.0
    if far_control_d_min is None:
        far_control_d_min = 2.0 * d_bar

    rng = np.random.default_rng(seed)

    # ~111 km per latitude degree on the equator (haversine approx).
    KM_PER_DEG = 111.195
    near_d_max_deg = near_control_d_max / KM_PER_DEG
    far_d_min_deg = far_control_d_min / KM_PER_DEG

    units = [f"u{i:04d}" for i in range(n_units)]
    mu = rng.normal(0.0, 0.5, size=n_units)
    # Time effects: linear trend
    lambda_t = np.array([0.05 * t for t in range(n_periods)])
    if parallel_trends_shock_sd > 0:
        lambda_t = lambda_t + rng.normal(0, parallel_trends_shock_sd, size=n_periods)

    coords = np.zeros((n_units, 2))
    is_treated = np.zeros(n_units, dtype=bool)
    is_near = np.zeros(n_units, dtype=bool)

    # Treated: cluster very near origin (within 1 km)
    for i in range(n_treated):
        coords[i, 0] = rng.normal(0.0, 0.005)
        coords[i, 1] = rng.normal(0.0, 0.005)
        is_treated[i] = True

    # Near-control: within near_d_max_deg of origin, but not at origin
    near_start = n_treated
    for i in range(n_near_control):
        idx = near_start + i
        # Uniformly distributed in an annulus to avoid clustering
        r_deg = rng.uniform(near_d_max_deg * 0.1, near_d_max_deg * 0.95)
        theta = rng.uniform(0, 2 * np.pi)
        coords[idx, 0] = r_deg * np.cos(theta)
        coords[idx, 1] = r_deg * np.sin(theta)
        is_near[idx] = True

    # Far-control: at far_d_min_deg+ from origin
    far_start = n_treated + n_near_control
    for i in range(n_far_control):
        idx = far_start + i
        r_deg = rng.uniform(far_d_min_deg, far_d_min_deg * 1.5)
        theta = rng.uniform(0, 2 * np.pi)
        coords[idx, 0] = r_deg * np.cos(theta)
        coords[idx, 1] = r_deg * np.sin(theta)

    rows = []
    for i, u in enumerate(units):
        ft = float(t_treat) if is_treated[i] else np.inf
        for t in range(n_periods):
            y_clean = mu[i] + lambda_t[t]
            post = t >= t_treat
            if is_treated[i] and post:
                y = y_clean + tau_total
            elif is_near[i] and post:
                y = y_clean + delta_1
            else:
                y = y_clean
            y += rng.normal(0, error_sd)
            rows.append(
                {
                    "unit": u,
                    "time": t,
                    "lat": coords[i, 0],
                    "lon": coords[i, 1],
                    "D": int(is_treated[i] and post),
                    "first_treat": ft,
                    "y": y,
                }
            )
    return pd.DataFrame(rows)


def generate_butts_staggered_dgp(
    *,
    n_cohorts: int = 3,
    units_per_cohort: int = 30,
    n_far_control: int = 90,
    n_periods: int = 6,
    cohort_onsets: Optional[list] = None,
    tau_total: float = -0.07,
    delta_1: float = -0.04,
    tau_per_event_time: Optional[Callable[[int], float]] = None,
    delta_per_ring_per_event_time: Optional[Callable[[int, int], float]] = None,
    d_bar: float = 100.0,
    error_sd: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """Staggered panel with known per-cohort effects.

    Three cohorts treat at distinct times; far-control group never treated.
    Each cohort is clustered at a distinct geographic center so spillover
    is well-defined per cohort (units near cohort C1 receive spillover
    starting at C1's onset, etc.).

    Parameters
    ----------
    tau_per_event_time : callable, optional
        If supplied, overrides scalar ``tau_total`` for treated rows. Called
        as ``tau_per_event_time(k)`` where ``k = t - first_treat[unit]`` is the
        event-time relative to own onset (``k >= 0`` for treated rows). Used
        by Wave C event-study identification regression tests to verify
        recovery of per-event-time direct effects.
    delta_per_ring_per_event_time : callable, optional
        If supplied, overrides scalar ``delta_1`` for spillover rows. Called
        as ``delta_per_ring_per_event_time(ring_idx, k)`` where ``ring_idx``
        is the spillover ring (0-indexed; this DGP places all near-controls
        in ring 0 by construction — one-cohort-one-cluster) and ``k`` is the
        spillover-exposure event-time (``t - spillover_onset_by_unit[unit]``,
        ``k >= 0`` for exposed rows).

    Notes
    -----
    When both callable kwargs default to ``None``, output is bit-identical to
    the pre-Wave-C baseline (asserted by ``tests/test_dgp_utils.py``).

    Returns the same column schema as :func:`generate_butts_nonstaggered_dgp`.
    """
    if cohort_onsets is None:
        cohort_onsets = [1, 2, 3][:n_cohorts]

    rng = np.random.default_rng(seed)
    KM_PER_DEG = 111.195

    rows = []
    coords_by_unit = {}
    first_treat_by_unit = {}
    is_treated_unit = {}
    # Spillover-trigger onset per near-control unit (the cohort it belongs to).
    spillover_onset_by_unit: dict = {}

    unit_idx = 0

    # Cohort centers spaced ~5*d_bar apart so cohorts don't overlap.
    cohort_spacing_deg = (5 * d_bar) / KM_PER_DEG

    for c_idx in range(n_cohorts):
        center_lat = c_idx * cohort_spacing_deg
        center_lon = 0.0
        onset = cohort_onsets[c_idx]
        for i in range(units_per_cohort):
            u = f"c{c_idx}_{unit_idx:04d}"
            unit_idx += 1
            # 1/3 of cohort members are "treated" (cluster at center);
            # 2/3 are "near-controls" within d_bar/2 of center.
            if i < units_per_cohort // 3:
                coords_by_unit[u] = (
                    center_lat + rng.normal(0, 0.003),
                    center_lon + rng.normal(0, 0.003),
                )
                first_treat_by_unit[u] = float(onset)
                is_treated_unit[u] = True
            else:
                r_deg = rng.uniform(0.05, (d_bar / 2.0) / KM_PER_DEG * 0.9)
                theta = rng.uniform(0, 2 * np.pi)
                coords_by_unit[u] = (
                    center_lat + r_deg * np.cos(theta),
                    center_lon + r_deg * np.sin(theta),
                )
                first_treat_by_unit[u] = np.inf
                is_treated_unit[u] = False
                # Near-control receives spillover when its cohort treats.
                spillover_onset_by_unit[u] = float(onset)

    # Far-control: placed far from all cohort centers
    far_center_lat = n_cohorts * cohort_spacing_deg + 5 * d_bar / KM_PER_DEG
    for i in range(n_far_control):
        u = f"far_{i:04d}"
        unit_idx += 1
        r_deg = rng.uniform(0.0, 1.0)
        theta = rng.uniform(0, 2 * np.pi)
        coords_by_unit[u] = (
            far_center_lat + r_deg * np.cos(theta),
            r_deg * np.sin(theta),
        )
        first_treat_by_unit[u] = np.inf
        is_treated_unit[u] = False
        # Far-controls receive no spillover (no nearby cohort).

    mu = {u: rng.normal(0, 0.5) for u in coords_by_unit}
    lambda_t = np.array([0.05 * t for t in range(n_periods)])

    for u, (lat, lon) in coords_by_unit.items():
        ft = first_treat_by_unit[u]
        is_treat = is_treated_unit[u]
        spillover_onset = spillover_onset_by_unit.get(u, np.inf)
        for t in range(n_periods):
            y_clean = mu[u] + lambda_t[t]
            # Direct effect for own treated unit
            if is_treat and t >= ft:
                if tau_per_event_time is not None:
                    k_direct = int(t - ft)
                    effect = tau_per_event_time(k_direct)
                else:
                    effect = tau_total
                y = y_clean + effect
            # Spillover on near-control once its cohort activates
            elif (not is_treat) and t >= spillover_onset:
                if delta_per_ring_per_event_time is not None:
                    k_spill = int(t - spillover_onset)
                    # ring_idx=0: this DGP places all near-controls in ring 0
                    # by construction (one cohort = one cluster center).
                    effect = delta_per_ring_per_event_time(0, k_spill)
                else:
                    effect = delta_1
                y = y_clean + effect
            else:
                y = y_clean
            y += rng.normal(0, error_sd)
            rows.append(
                {
                    "unit": u,
                    "time": t,
                    "lat": lat,
                    "lon": lon,
                    "D": int(is_treat and t >= ft),
                    "first_treat": ft,
                    "y": y,
                }
            )
    return pd.DataFrame(rows)


def make_lpdid_panel(
    *,
    cohorts=(5, 8),
    n_per_cohort: int = 20,
    n_never: int = 30,
    n_periods: int = 12,
    tau: Callable[[int], float] = lambda k: 1.0 + 0.5 * k,
    unit_fe_sd: float = 0.5,
    time_trend: float = 0.5,
    error_sd: float = 0.2,
    heterogeneous: bool = True,
    seed: int = 20260628,
) -> pd.DataFrame:
    """Staggered, absorbing-treatment panel for LP-DiD tests.

    Emits columns the two estimator families both need:

    - ``treat``: binary absorbing indicator ``1[t >= first_treat]`` (LPDiD's
      ``treatment=`` input; never-treated units are always 0).
    - ``first_treat``: onset period; ``np.inf`` for never-treated (the repo
      convention consumed by CallawaySantAnna / ImputationDiD / StackedDiD).

    Untreated potential outcome is ``alpha_i + time_trend * t + noise`` so
    parallel trends and no-anticipation hold by construction. The dynamic
    effect at event time ``k = t - g`` is ``tau(k)`` for ``k >= 0`` (zero
    otherwise), scaled per cohort when ``heterogeneous=True`` so different
    cohorts have genuinely different effect paths.
    """
    rng = np.random.default_rng(seed)
    rows = []
    uid = 0
    for ci, g in enumerate(cohorts):
        scale = (1.0 + 0.3 * ci) if heterogeneous else 1.0
        for _ in range(n_per_cohort):
            uid += 1
            alpha = rng.normal(0.0, unit_fe_sd)
            for t in range(n_periods):
                y0 = alpha + time_trend * t + rng.normal(0.0, error_sd)
                k = t - g
                effect = scale * tau(k) if k >= 0 else 0.0
                rows.append(
                    {
                        "unit": uid,
                        "time": t,
                        "y": y0 + effect,
                        "treat": int(t >= g),
                        "first_treat": float(g),
                    }
                )
    for _ in range(n_never):
        uid += 1
        alpha = rng.normal(0.0, unit_fe_sd)
        for t in range(n_periods):
            y0 = alpha + time_trend * t + rng.normal(0.0, error_sd)
            rows.append(
                {
                    "unit": uid,
                    "time": t,
                    "y": y0,
                    "treat": 0,
                    "first_treat": np.inf,
                }
            )
    return pd.DataFrame(rows)
