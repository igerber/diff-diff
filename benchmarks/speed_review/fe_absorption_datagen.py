"""Deterministic DGPs for the FE-absorption benchmark suite.

Shared by ``bench_fe_absorption.py`` (diff-diff lane) and
``bench_fe_absorption_pyfixest.py`` (optional external yardstick). Both lanes
regenerate data locally from the same seeded PCG64 stream, so no data files
cross process boundaries; each scenario records a checksum of the outcome
column so the driver can prove both lanes saw identical matrices.

Scenario shapes are calibrated to practitioner workloads, not to flatter the
library (see docs/performance-scenarios.md, FE-absorption suite section, for
the provenance of each):

- ``county_policy``   3,109 counties (the US county count) x 60 months,
                      staggered adoption, ~5% attrition - the everyday
                      applied-micro event study.
- ``firm_churn``      100k firms x 40 quarters with contiguous entry/exit
                      lifetimes (~60% of the window) - Compustat/LBD-style
                      churn; correlated FE incidence is the slow regime for
                      alternating projections.
- ``scanner_twfe``    35k stores x 156 weeks with entry/exit - Nielsen-scanner
                      shape for the TWFE workhorse.
- ``geo_experiment``  5M order-level rows, 100k stores + 1k weeks - the
                      static-geo experiment shape from the Instacart pyfixest
                      writeup (tech.instacart.com, 2025).
- ``survey_absorb``   500k microdata rows, state+month FE, 80 BRR replicate
                      weights - the replicate-refit multiplier path (each
                      replicate re-runs the weighted demean).
- ``tail_stress``     5M rows with contiguous ~20% store lifetimes - the
                      correlated-FE stress regime (~250+ MAP iterations).
                      Reported separately; NOT a headline scenario.
- ``guard_small``     balanced 1k x 20 panel - regression guard so the small-
                      data path never regresses.
"""

import numpy as np
import pandas as pd

TAU = 0.25  # true treatment effect in every DGP


def checksum(df, col="y"):
    return float(np.asarray(df[col], dtype=np.float64).sum())


# --------------------------------------------------------------------- panels
def make_panel(n_units, n_periods, seed=42, drop_frac=0.10, lifetime_span=None):
    """2x2 block-adoption panel: half the units treated after the midpoint.

    drop_frac > 0 drops rows at random (mild unbalance). lifetime_span in
    (0, 1] instead gives each unit a contiguous observation window covering
    that fraction of the periods (entry/exit churn - correlated FE incidence).
    """
    rng = np.random.default_rng(seed)
    unit = np.repeat(np.arange(n_units, dtype=np.int64), n_periods)
    time = np.tile(np.arange(n_periods, dtype=np.int64), n_units)
    alpha = rng.normal(0.0, 1.0, n_units)
    gamma = rng.normal(0.0, 0.5, n_periods)
    treated = (unit < n_units // 2).astype(np.int8)
    post = (time >= n_periods // 2).astype(np.int8)
    d = (treated * post).astype(np.int8)
    y = alpha[unit] + gamma[time] + TAU * d + rng.normal(0.0, 1.0, unit.size)
    df = pd.DataFrame(
        {"y": y, "treated": treated, "post": post, "d": d, "unit": unit, "time": time}
    )
    if lifetime_span is not None:
        span = max(2, int(n_periods * lifetime_span))
        entry = rng.integers(0, n_periods - span + 1, n_units)
        keep = (time >= entry[unit]) & (time < entry[unit] + span)
        df = df.loc[keep].reset_index(drop=True)
    elif drop_frac:
        keep = rng.random(len(df)) >= drop_frac
        df = df.loc[keep].reset_index(drop=True)
    return df


def make_staggered_panel(n_units, n_periods, cohorts, seed=42, drop_frac=0.0, lifetime_span=None):
    """Staggered-adoption panel for SunAbraham: cohorts = adoption periods
    (0 entries mean never-treated). Equal cohort shares including never.
    Dynamic effect ramps with event time (heterogeneous-by-cohort trends are
    not needed to exercise the demeaning machinery).
    """
    rng = np.random.default_rng(seed)
    unit = np.repeat(np.arange(n_units, dtype=np.int64), n_periods)
    time = np.tile(np.arange(n_periods, dtype=np.int64), n_units)
    choices = np.asarray(list(cohorts) + [0], dtype=np.int64)
    g_unit = rng.choice(choices, n_units)
    first = g_unit[unit]
    rel = time - first
    d = ((first > 0) & (rel >= 0)).astype(np.int8)
    alpha = rng.normal(0.0, 1.0, n_units)
    gamma = rng.normal(0.0, 0.5, n_periods)
    effect = np.where(d == 1, TAU * (1.0 + 0.05 * np.clip(rel, 0, 10)), 0.0)
    y = alpha[unit] + gamma[time] + effect + rng.normal(0.0, 1.0, unit.size)
    df = pd.DataFrame({"y": y, "unit": unit, "time": time, "first_treat": first, "d": d})
    if lifetime_span is not None:
        span = max(2, int(n_periods * lifetime_span))
        entry = rng.integers(0, n_periods - span + 1, n_units)
        t = df["time"].values
        u = df["unit"].values
        keep = (t >= entry[u]) & (t < entry[u] + span)
        df = df.loc[keep].reset_index(drop=True)
    elif drop_frac:
        keep = rng.random(len(df)) >= drop_frac
        df = df.loc[keep].reset_index(drop=True)
    return df


def make_orders(n_rows, n_stores, n_weeks, seed=42, lifetime_span=None):
    """Order-level static-geo experiment: store + week FE, store-level
    treatment switched on for the post half of the weeks. lifetime_span
    gives each store a contiguous active week window (the correlated-FE
    regime); None gives uniform random incidence (the experiment regime).
    """
    rng = np.random.default_rng(seed)
    store = rng.integers(0, n_stores, n_rows).astype(np.int64)
    if lifetime_span is not None:
        span = max(2, int(n_weeks * lifetime_span))
        entry = rng.integers(0, n_weeks - span + 1, n_stores)
        week = (entry[store] + rng.integers(0, span, n_rows)).astype(np.int64)
    else:
        week = rng.integers(0, n_weeks, n_rows).astype(np.int64)
    a = rng.normal(0.0, 1.0, n_stores)
    b = rng.normal(0.0, 0.5, n_weeks)
    treated = (rng.random(n_stores) < 0.5).astype(np.int8)[store]
    post = (week >= n_weeks // 2).astype(np.int8)
    d = (treated * post).astype(np.int8)
    y = a[store] + b[week] + TAU * d + rng.normal(0.0, 1.0, n_rows)
    return pd.DataFrame(
        {"y": y, "treated": treated, "post": post, "d": d, "store": store, "week": week}
    )


def make_survey_absorb(n_rows, n_states, n_months, n_replicates, seed=42):
    """State-policy microdata with sampling weights and BRR replicate weights.

    Half the states adopt at the month midpoint. Replicate weights follow the
    standard BRR half-sample convention (w * 2 on the selected half, w * 0 on
    the other), supplied as pre-computed columns rw1..rwK the way BRFSS-style
    public-use files ship them.
    """
    rng = np.random.default_rng(seed)
    state = rng.integers(0, n_states, n_rows).astype(np.int64)
    month = rng.integers(0, n_months, n_rows).astype(np.int64)
    a = rng.normal(0.0, 1.0, n_states)
    b = rng.normal(0.0, 0.3, n_months)
    treated = (state < n_states // 2).astype(np.int8)
    post = (month >= n_months // 2).astype(np.int8)
    d = (treated * post).astype(np.int8)
    y = a[state] + b[month] + TAU * d + rng.normal(0.0, 1.0, n_rows)
    w = rng.lognormal(0.0, 0.5, n_rows)
    cols = {
        "y": y,
        "treated": treated,
        "post": post,
        "d": d,
        "state": state,
        "month": month,
        "w": w,
    }
    # BRR half-samples drawn at the state level (states act as the PSUs).
    for r in range(1, n_replicates + 1):
        half = rng.random(n_states) < 0.5
        cols[f"rw{r}"] = w * np.where(half[state], 2.0, 0.0)
    return pd.DataFrame(cols)


# ----------------------------------------------------------- scenario registry
def build(scenario, quick=False, seed=42):
    """Return (DataFrame, meta) for a scenario id. quick=True shrinks every
    dimension ~100x for smoke runs; the committed baselines use quick=False.
    """
    q = quick
    if scenario == "county_policy":
        df = make_staggered_panel(
            n_units=300 if q else 3_109,
            n_periods=12 if q else 60,
            cohorts=(6, 8) if q else (24, 30, 36, 42),
            seed=seed,
            drop_frac=0.05,
        )
    elif scenario == "firm_churn":
        df = make_staggered_panel(
            n_units=1_000 if q else 100_000,
            n_periods=12 if q else 40,
            cohorts=(6, 8) if q else (16, 20, 24, 28),
            seed=seed,
            lifetime_span=0.6,
        )
    elif scenario == "scanner_twfe":
        df = make_panel(
            n_units=500 if q else 35_000,
            n_periods=12 if q else 156,
            seed=seed,
            lifetime_span=0.6,
        )
    elif scenario == "geo_experiment":
        df = make_orders(
            n_rows=50_000 if q else 5_000_000,
            n_stores=1_000 if q else 100_000,
            n_weeks=100 if q else 1_000,
            seed=seed,
        )
    elif scenario == "survey_absorb":
        df = make_survey_absorb(
            n_rows=5_000 if q else 500_000,
            n_states=50,
            n_months=12 if q else 60,
            n_replicates=8 if q else 80,
            seed=seed,
        )
    elif scenario == "tail_stress":
        df = make_orders(
            n_rows=50_000 if q else 5_000_000,
            n_stores=1_000 if q else 100_000,
            n_weeks=100 if q else 1_000,
            seed=seed,
            lifetime_span=0.2,
        )
    elif scenario == "guard_small":
        df = make_panel(n_units=1_000, n_periods=20, seed=seed, drop_frac=0.0)
    else:
        raise ValueError(f"unknown scenario {scenario!r}")
    meta = {"n_obs": int(len(df)), "checksum": checksum(df)}
    return df, meta


SCENARIO_IDS = (
    "county_policy",
    "firm_churn",
    "scanner_twfe",
    "geo_experiment",
    "survey_absorb",
    "tail_stress",
    "guard_small",
)
