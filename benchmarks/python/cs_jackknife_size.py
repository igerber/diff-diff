#!/usr/bin/env python3
"""Size of CallawaySantAnna inference with few clusters: hc1, bootstrap, jackknife.

Question: at a nominal 5% level and with no treatment effect, how often do
the three inference methods of ``CallawaySantAnna`` reject, as a function of
the number of clusters and of treated clusters? Karim, Nielsen, MacKinnon &
Webb (2026, arXiv:2602.12043) answer it for Stata ``csdid`` on CPS data
(Tables 2-3). This script answers it for this library, whose clustered
``hc1`` path already differs from ``csdid`` (a ``G/(G-1)`` factor and a
``t(G-1)`` reference), and it covers the quantities the paper has no
evidence for: event-study coefficients and the pre-trend test.

Everything below was fixed before the first run and is not searched.

Design (the paper's): ``T = 8`` periods, two cohorts adopting in periods 4
and 6 with ``J`` treated clusters each, the remaining clusters never
treated, no effect.

Outcome: ``y_it = a_c + d_t + u_ct + e_it``, every draw Gaussian. ``a_c`` and
``d_t`` are ``N(0, 1)``. ``u_ct`` is a cluster-level AR(1) with coefficient
0.8 and innovation sd 1, started from its stationary distribution
``N(0, 1 / (1 - 0.64))``. ``e_it`` is unit-level iid ``N(0, 1)``, present
only when a cluster has more than one unit.

Grid: the paper's Table 1 (21 cells, each leaving at least two never-treated
clusters) for 1 and for 5 units per cluster, 42 design points. 1,000
replications per design point; the design points' generators are
``np.random.SeedSequence(20261125).spawn(42)`` in grid order, so the results
do not depend on how the points are scheduled.

Methods, all clustered on the cluster column:
  hc1        ``CallawaySantAnna(cluster=...)``
  bootstrap  ``CallawaySantAnna(cluster=..., n_bootstrap=499)``
  jackknife  ``CallawaySantAnna(cluster=..., vcov_type="jackknife")``

Quantities: the overall ATT; each post-treatment event-study coefficient
(event times 0 to 4); the pre-period coefficient at event time -2; and two
pre-trend tests over the pre-period event-study coefficients: Bonferroni on
the per-row p-values (the test ``DiagnosticReport`` applies to a jackknife
fit) and the chi-square joint Wald on the event-study covariance (the test
it applies to an ``hc1`` fit).

Scoring: a replication whose inference is undefined (NaN p-value) is counted
on its own and never as a non-rejection; rejection and 95% coverage are over
the defined replications.

Flag rule (set in advance; it flags, it hides nothing): a design point is
flagged when the jackknife rejection rate of an event-study coefficient or
of the Bonferroni pre-trend test exceeds the jackknife rejection rate of the
overall ATT at the same design point by more than 3 percentage points. With
1,000 replications a difference of two such rates has a Monte Carlo standard
error near 0.01, and about 300 comparisons are made, so a few flags are
expected by chance alone; the count is to be read against that.

The output JSON underwrites the "Cluster jackknife" notes in
``docs/methodology/REGISTRY.md`` section ``CallawaySantAnna``.

Usage:
    python benchmarks/python/cs_jackknife_size.py \\
        --output benchmarks/data/cs_jackknife_size.json

    # Timing only: 20 replications of the largest design point
    python benchmarks/python/cs_jackknife_size.py --timing
"""

import argparse
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "MKL_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import diff_diff  # noqa: E402
from diff_diff import CallawaySantAnna  # noqa: E402

SEED = 20261125
ALPHA = 0.05
N_PERIODS = 8
COHORTS = (4, 6)
RHO = 0.8
N_BOOTSTRAP = 499
POST_EVENT_TIMES = (0, 1, 2, 3, 4)
PRE_EVENT_TIME = -2
FLAG_MARGIN = 0.03
TABLE_1 = {
    8: (1, 2, 3),
    16: (1, 2, 3, 4),
    24: (1, 2, 3, 4, 6, 8),
    32: (1, 2, 3, 4, 6, 8, 10, 12),
}
UNITS_PER_CLUSTER = (1, 5)
METHODS = ("hc1", "bootstrap", "jackknife")
FIT_KW = {"outcome": "y", "unit": "unit", "time": "t", "first_treat": "g"}


def design_points():
    """The 42 design points, in the fixed order their seeds are spawned in."""
    return [(h, j, m) for m in UNITS_PER_CLUSTER for h in sorted(TABLE_1) for j in TABLE_1[h]]


def simulate_panel(rng, n_clusters, n_treated_per_cohort, units_per_cluster):
    """One panel of the fixed design with no treatment effect."""
    periods = np.arange(1, N_PERIODS + 1)
    cluster_effect = rng.normal(size=n_clusters)
    period_effect = rng.normal(size=N_PERIODS)
    shock = np.empty((n_clusters, N_PERIODS))
    shock[:, 0] = rng.normal(scale=np.sqrt(1.0 / (1.0 - RHO**2)), size=n_clusters)
    for t in range(1, N_PERIODS):
        shock[:, t] = RHO * shock[:, t - 1] + rng.normal(size=n_clusters)
    cohort = np.zeros(n_clusters, dtype=int)
    cohort[:n_treated_per_cohort] = COHORTS[0]
    cohort[n_treated_per_cohort : 2 * n_treated_per_cohort] = COHORTS[1]

    cluster = np.repeat(np.arange(n_clusters), units_per_cluster)
    n_units = cluster.size
    y = cluster_effect[cluster][:, None] + period_effect[None, :] + shock[cluster]
    if units_per_cluster > 1:
        y = y + rng.normal(size=(n_units, N_PERIODS))
    return pd.DataFrame(
        {
            "unit": np.repeat(np.arange(n_units), N_PERIODS),
            "cluster": np.repeat(cluster, N_PERIODS),
            "t": np.tile(periods, n_units),
            "g": np.repeat(cohort[cluster], N_PERIODS),
            "y": y.reshape(-1),
        }
    )


def estimator(method, seed):
    if method == "hc1":
        return CallawaySantAnna(cluster="cluster")
    if method == "bootstrap":
        return CallawaySantAnna(cluster="cluster", n_bootstrap=N_BOOTSTRAP, seed=seed)
    return CallawaySantAnna(cluster="cluster", vcov_type="jackknife")


def quantities(results):
    """``{name: (p_value, covers_zero)}`` for one fit; NaN p marks undefined."""
    out = {
        "overall": (
            float(results.p_value),
            bool(results.conf_int[0] <= 0.0 <= results.conf_int[1]),
        )
    }
    es = results.aggregate("event_study")
    times = [int(e) for e in np.asarray(es.event_time).tolist()]
    position = {e: i for i, e in enumerate(times)}
    for e in POST_EVENT_TIMES + (PRE_EVENT_TIME,):
        name = f"es_{e}"
        if e not in position:
            out[name] = (float("nan"), False)
            continue
        i = position[e]
        out[name] = (
            float(es.p_value[i]),
            bool(es.conf_int_lower[i] <= 0.0 <= es.conf_int_upper[i]),
        )

    pre = [i for i, e in enumerate(times) if e < 0 and not bool(es.is_reference[i])]
    pre_p = np.array([float(es.p_value[i]) for i in pre])
    # Bonferroni over the per-row p-values, valid only when every retained
    # pre-period row has one (the report's rule).
    if pre and np.all(np.isfinite(pre_p)):
        out["pretrend_bonferroni"] = (float(min(1.0, len(pre) * pre_p.min())), False)
    else:
        out["pretrend_bonferroni"] = (float("nan"), False)
    # Chi-square joint Wald on the event-study covariance.
    wald_p = float("nan")
    if es.vcov is not None and es.vcov_index is not None and pre:
        index = [int(e) for e in np.asarray(es.vcov_index).tolist()]
        where = {e: k for k, e in enumerate(index)}
        pre_times = [times[i] for i in pre]
        if all(e in where for e in pre_times):
            rows = [where[e] for e in pre_times]
            sub = np.asarray(es.vcov)[np.ix_(rows, rows)]
            beta = np.array([float(es.att[i]) for i in pre])
            try:
                stat = float(beta @ np.linalg.solve(sub, beta))
                if np.isfinite(stat) and stat >= 0:
                    wald_p = float(stats.chi2.sf(stat, len(pre)))
            except np.linalg.LinAlgError:
                pass
    out["pretrend_wald_chi2"] = (wald_p, False)
    return out


def run_design_point(args):
    """All replications of one design point. Returns its result record."""
    index, n_clusters, n_treated, units_per_cluster, n_reps, child_state = args
    rng = np.random.default_rng(np.random.SeedSequence(**child_state))
    tallies = {}
    failures = {method: 0 for method in METHODS}
    started = time.perf_counter()
    for _ in range(n_reps):
        panel = simulate_panel(rng, n_clusters, n_treated, units_per_cluster)
        boot_seed = int(rng.integers(0, 2**31 - 1))
        for method in METHODS:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit = estimator(method, boot_seed).fit(panel, **FIT_KW)
                    values = quantities(fit)
            except Exception:  # noqa: BLE001 - counted, never swallowed silently
                failures[method] += 1
                continue
            for name, (p_value, covers) in values.items():
                cell = tallies.setdefault((method, name), [0, 0, 0, 0])
                cell[0] += 1
                if np.isfinite(p_value):
                    cell[1] += 1
                    cell[2] += int(p_value < ALPHA)
                    cell[3] += int(covers)
    record = {
        "index": index,
        "n_clusters": n_clusters,
        "n_treated_per_cohort": n_treated,
        "units_per_cluster": units_per_cluster,
        "n_replications": n_reps,
        "fit_failures": failures,
        "seconds": round(time.perf_counter() - started, 1),
        "results": {},
    }
    for (method, name), (n_fit, n_defined, n_reject, n_cover) in sorted(tallies.items()):
        rate = n_reject / n_defined if n_defined else float("nan")
        record["results"].setdefault(method, {})[name] = {
            "n_fit": n_fit,
            "n_defined": n_defined,
            "share_undefined": 1.0 - n_defined / n_fit if n_fit else float("nan"),
            "rejection_rate": rate,
            "rejection_mc_se": (
                float(np.sqrt(rate * (1.0 - rate) / n_defined)) if n_defined else float("nan")
            ),
            "coverage": (
                n_cover / n_defined
                if n_defined and not name.startswith("pretrend")
                else float("nan")
            ),
        }
    return record


def flags(record):
    """Jackknife quantities whose rejection exceeds the overall ATT's by the margin."""
    jk = record["results"].get("jackknife", {})
    base = jk.get("overall", {}).get("rejection_rate", float("nan"))
    out = []
    for name, cell in jk.items():
        if name in ("overall", "pretrend_wald_chi2"):
            continue
        rate = cell["rejection_rate"]
        if np.isfinite(rate) and np.isfinite(base) and rate - base > FLAG_MARGIN:
            out.append({"quantity": name, "rejection_rate": rate, "overall": base})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--n-replications", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--timing", action="store_true", help="time 20 replications, no results")
    args = parser.parse_args()

    points = design_points()
    children = np.random.SeedSequence(SEED).spawn(len(points))
    jobs = [
        (
            i,
            h,
            j,
            m,
            args.n_replications,
            {"entropy": child.entropy, "spawn_key": child.spawn_key},
        )
        for i, ((h, j, m), child) in enumerate(zip(points, children))
    ]

    if args.timing:
        i, h, j, m, _, state = max(jobs, key=lambda job: (job[1] * job[3], job[2]))
        started = time.perf_counter()
        run_design_point((i, h, j, m, 20, state))
        per_rep = (time.perf_counter() - started) / 20
        print(
            f"largest design point (H={h}, J={j}, units/cluster={m}): {per_rep:.3f} s/replication"
        )
        return

    started = time.perf_counter()
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for record in pool.map(run_design_point, jobs):
            record["flags"] = flags(record)
            records.append(record)
            jk = record["results"]["jackknife"]["overall"]["rejection_rate"]
            hc1 = record["results"]["hc1"]["overall"]["rejection_rate"]
            print(
                f"H={record['n_clusters']:>2} J={record['n_treated_per_cohort']:>2} "
                f"m={record['units_per_cluster']}: overall hc1 {hc1:.3f} jackknife {jk:.3f} "
                f"flags {len(record['flags'])} ({record['seconds']} s)",
                flush=True,
            )
    payload = {
        "meta": {
            "seed": SEED,
            "alpha": ALPHA,
            "n_periods": N_PERIODS,
            "cohorts": list(COHORTS),
            "ar1_coefficient": RHO,
            "n_bootstrap": N_BOOTSTRAP,
            "n_replications": args.n_replications,
            "flag_margin": FLAG_MARGIN,
            "diff_diff_version": diff_diff.__version__,
            "backend": os.environ.get("DIFF_DIFF_BACKEND", "auto"),
            "wall_seconds": round(time.perf_counter() - started, 1),
            "workers": args.workers,
        },
        "design_points": sorted(records, key=lambda r: r["index"]),
    }
    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
