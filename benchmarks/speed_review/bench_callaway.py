"""
Benchmark CallawaySantAnna.fit() at multiple scales with per-phase granularity.

Usage:
    python benchmarks/speed_review/bench_callaway.py
"""

import time
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from diff_diff import CallawaySantAnna


def generate_staggered_data(n_units, n_periods=10, n_cohorts=5, seed=42):
    """Generate panel data with staggered treatment adoption."""
    rng = np.random.default_rng(seed)

    # Assign cohorts: ~20% never-treated, rest split among cohorts
    treatment_periods = np.linspace(3, n_periods - 2, n_cohorts, dtype=int)
    cohort_assignment = rng.choice(
        [0] + list(treatment_periods),
        size=n_units,
        p=[0.2] + [0.8 / n_cohorts] * n_cohorts,
    )

    rows = []
    for i in range(n_units):
        g = cohort_assignment[i]
        for t in range(1, n_periods + 1):
            treated = 1 if (g > 0 and t >= g) else 0
            y = rng.normal(0, 1) + 2.0 * treated
            rows.append((i, t, y, g))

    df = pd.DataFrame(rows, columns=["unit", "time", "outcome", "first_treat"])
    return df


def bench_fit(n_units, n_bootstrap=0, covariates=None, n_cohorts=5, n_runs=3,
              estimation_method="reg"):
    """Benchmark fit() and return median time."""
    df = generate_staggered_data(n_units, n_cohorts=n_cohorts)

    if covariates:
        rng = np.random.default_rng(99)
        for cov in covariates:
            df[cov] = rng.normal(size=len(df))

    cs = CallawaySantAnna(
        n_bootstrap=n_bootstrap,
        seed=123,
        estimation_method=estimation_method,
    )

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        res = cs.fit(
            df,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            covariates=covariates,
        )
        # Keep "fit + aggregations" as the measured work (the 3.x
        # fit(aggregate="all") route, deprecated since 3.9 and removed at the 4.0 cut).
        res.aggregate("event_study")
        res.aggregate("group")
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return np.median(times)


def main():
    scales = [1_000, 5_000, 10_000, 50_000]
    print("=" * 72)
    print("CallawaySantAnna Benchmark Suite")
    print("=" * 72)

    # No-covariates, no bootstrap
    print("\n--- No covariates, no bootstrap ---")
    print(f"{'Units':>10}  {'Time (s)':>10}")
    for n in scales:
        t = bench_fit(n, n_bootstrap=0, n_runs=3)
        print(f"{n:>10}  {t:>10.4f}")

    # No-covariates, with bootstrap
    print("\n--- No covariates, bootstrap=999 ---")
    print(f"{'Units':>10}  {'Time (s)':>10}")
    for n in scales[:3]:  # skip 50K with bootstrap (too slow)
        t = bench_fit(n, n_bootstrap=999, n_runs=1)
        print(f"{n:>10}  {t:>10.4f}")

    # With covariates, no bootstrap (reg)
    print("\n--- 2 covariates, reg, no bootstrap ---")
    print(f"{'Units':>10}  {'Time (s)':>10}")
    for n in scales[:3]:
        t = bench_fit(n, n_bootstrap=0, covariates=["x1", "x2"], n_runs=3)
        print(f"{n:>10}  {t:>10.4f}")

    # With 10 covariates, no bootstrap (reg)
    cov10 = [f"x{i}" for i in range(1, 11)]
    print("\n--- 10 covariates, reg, no bootstrap ---")
    print(f"{'Units':>10}  {'Time (s)':>10}")
    for n in scales[:3]:
        t = bench_fit(n, n_bootstrap=0, covariates=cov10, n_runs=3)
        print(f"{n:>10}  {t:>10.4f}")

    # With 2 covariates, DR, no bootstrap
    print("\n--- 2 covariates, dr, no bootstrap ---")
    print(f"{'Units':>10}  {'Time (s)':>10}")
    for n in scales[:3]:
        t = bench_fit(n, n_bootstrap=0, covariates=["x1", "x2"], n_runs=3,
                      estimation_method="dr")
        print(f"{n:>10}  {t:>10.4f}")

    # With 2 covariates, IPW, no bootstrap
    print("\n--- 2 covariates, ipw, no bootstrap ---")
    print(f"{'Units':>10}  {'Time (s)':>10}")
    for n in scales[:3]:
        t = bench_fit(n, n_bootstrap=0, covariates=["x1", "x2"], n_runs=3,
                      estimation_method="ipw")
        print(f"{n:>10}  {t:>10.4f}")

    # With 10 covariates, 50K units (reg)
    print("\n--- 10 covariates, reg, 50K units ---")
    t = bench_fit(50_000, n_bootstrap=0, covariates=cov10, n_runs=1)
    print(f"{'50000':>10}  {t:>10.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
