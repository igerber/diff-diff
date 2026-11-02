"""Frozen-contract follow-up and controlled calibration experiments for ChainedDiD.

The oracle is validation-only: it knows population treatment odds and/or cohort
masses. It is not an estimator option. Original qualification files are read-only.
"""

import argparse
import hashlib
import json
import platform
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from benchmarks.python.validate_chained_did import generate_data
from benchmarks.python.validate_chained_did import summarize as base_summary
from diff_diff import ChainedDiD

BASELINE = Path("benchmarks/data/chained_did/qualification.json")
TARGETS = ("overall", "g2_t3", "g3_t4")
VARIANTS = ("fitted_empirical", "fitted_known_mass", "oracle_empirical", "oracle_known_mass")


def replication_seed(seed, case_index, replication):
    """Use a fresh 64-bit seed stream without the original 32-bit truncation."""
    return int(
        np.random.SeedSequence([seed, case_index, replication]).generate_state(1, dtype=np.uint64)[
            0
        ]
    )


def diagnostic_cases():
    """Cross n and K rather than attributing their joint change to either one."""
    return [
        dict(n=n, observation=observation, clusters=k, null=False, violation="none")
        for n in (1000, 4000)
        for observation, k in (
            ("rotating", 0),
            ("intermittent", 0),
            ("intermittent", 50),
            ("intermittent", 100),
        )
    ]


def population_law():
    x = np.array([0.0, 1.0])
    p = np.column_stack([np.ones(2), np.exp(-0.2 + 0.7 * x), np.exp(-0.4 - 0.4 * x)])
    p /= p.sum(axis=1)[:, None]
    return {2: float(p[:, 1].mean()), 3: float(p[:, 2].mean())}


def oracle_cells(data, odds_by_cohort):
    """Independently recover post cells using known odds and ratio-score IFs.

    Cohort-specific pair response rates multiply odds by a constant, which
    cancels in the normalised control mean. These diagnostics use the valid
    baseline sampling designs only. Full rank is required, not presumed.
    """
    registry = data.drop_duplicates("id")
    times = sorted(data.t.unique())
    y = (
        data.pivot(index="id", columns="t", values="y")
        .reindex(index=registry.id, columns=times)
        .to_numpy()
    )
    groups = registry.g.to_numpy()
    cells, influence = {}, {}
    for g, odds in odds_by_cohort.items():
        ref = times.index(g) - 1
        columns = [j for j in range(len(times)) if j != ref]
        rows, contrasts, scores = [], [], []
        for s in range(len(times)):
            for t in range(s + 1, len(times)):
                treated = (groups == g) & np.isfinite(y[:, s]) & np.isfinite(y[:, t])
                control = (groups == 0) & np.isfinite(y[:, s]) & np.isfinite(y[:, t])
                if not treated.any() or not control.any():
                    continue
                delta = np.nan_to_num(y[:, t] - y[:, s], nan=0.0)
                w = control * odds
                mt, mc = delta[treated].mean(), w @ delta / w.sum()
                q = treated * (delta - mt) / treated.sum() - w * (delta - mc) / w.sum()
                row = np.zeros(len(times))
                row[s], row[t] = -1, 1
                rows.append(row[columns])
                contrasts.append(mt - mc)
                scores.append(q)
        a = np.asarray(rows)
        if not rows or np.linalg.matrix_rank(a) != len(columns):
            raise ValueError("Oracle diagnostic requires a connected comparison system")
        recovery = np.linalg.lstsq(a, np.eye(len(rows)), rcond=None)[0]
        estimates = recovery @ contrasts
        contributions = np.column_stack(scores) @ recovery.T
        for j, t in enumerate(columns):
            if times[t] >= g:
                cells[g, times[t]] = float(estimates[j])
                influence[g, times[t]] = contributions[:, j]
    return cells, influence


def combine(cells, influence, membership, masses=None):
    """Combine cell contributions with either estimated or known population mass."""
    keys = list(cells)
    effects = np.array([cells[k] for k in keys])
    indicators = np.column_stack([membership == g for g, _ in keys]).astype(float)
    mass = indicators.sum(axis=0) if masses is None else np.array([masses[g] for g, _ in keys])
    weights = mass / mass.sum()
    q = np.column_stack([influence[k] for k in keys]) @ weights
    if masses is None:
        # Ratio differentiation retains uncertainty in the supplied registry shares.
        estimate = weights @ effects
        q += indicators @ (effects - estimate) / mass.sum()
    return float(weights @ effects), q


def standard_error(q, labels=None):
    if labels is None:
        return float(np.linalg.norm(q))
    codes, unique = pd.factorize(labels, sort=False)
    if len(unique) < 2:
        return np.nan
    sums = np.zeros(len(unique))
    np.add.at(sums, codes, q)
    return float(np.sqrt(len(unique) / (len(unique) - 1) * (sums @ sums)))


def summarize(estimates, ses, truth):
    result = base_summary(estimates, ses, truth)
    valid = np.isfinite(estimates) & np.isfinite(ses) & (ses > 0)
    z = (estimates[valid] - truth) / ses[valid]
    result.update(
        z_mean=float(z.mean()) if len(z) else np.nan,
        z_sd=float(z.std(ddof=1)) if len(z) > 1 else np.nan,
        z_quantiles=np.quantile(z, [0.025, 0.5, 0.975]).tolist() if len(z) else [np.nan] * 3,
        abs_z_95=float(np.quantile(np.abs(z), 0.95)) if len(z) else np.nan,
    )
    return result


def run_case(case, replications, seed, index, diagnostic):
    started = time.monotonic()
    variants = VARIANTS if diagnostic else VARIANTS[:1]
    estimates = np.full((replications, len(variants), 3), np.nan)
    ses = np.full_like(estimates, np.nan)
    failures = {name: {} for name in variants}
    for j in range(replications):
        data, truth = generate_data(seed=replication_seed(seed, index, j), **case)
        registry = data.drop_duplicates("id")
        membership = registry.g.to_numpy()
        labels = registry.cluster.to_numpy() if case["clusters"] else None
        fitted = None
        try:
            fitted = ChainedDiD(cluster="cluster" if labels is not None else None).fit(
                data, "y", "id", "t", "g", covariates=["x"]
            )
        except (ValueError, np.linalg.LinAlgError) as exc:
            for name in variants[:2]:
                failures[name][str(exc)] = failures[name].get(str(exc), 0) + 1
        oracle = None
        if diagnostic:
            try:
                x = registry.x.to_numpy()
                oracle = oracle_cells(data, {2: np.exp(-0.2 + 0.7 * x), 3: np.exp(-0.4 - 0.4 * x)})
            except (ValueError, np.linalg.LinAlgError) as exc:
                for name in variants[2:]:
                    failures[name][str(exc)] = failures[name].get(str(exc), 0) + 1
        for v, name in enumerate(variants):
            if name.startswith("fitted"):
                if fitted is None:
                    continue
                cells = {
                    k: cell.effect for k, cell in fitted.group_time_effects.items() if k[1] >= k[0]
                }
                q = dict(zip(fitted.vcov_index, fitted.influence_functions.T))
            else:
                if oracle is None:
                    continue
                cells, q = oracle
            masses = population_law() if name.endswith("known_mass") else None
            overall, overall_q = combine(cells, q, membership, masses)
            estimates[j, v] = [overall, cells[2, 3], cells[3, 4]]
            ses[j, v] = [
                standard_error(overall_q, labels),
                standard_error(q[2, 3], labels),
                standard_error(q[3, 4], labels),
            ]
    return dict(
        design=case,
        elapsed_seconds=time.monotonic() - started,
        variants={
            name: dict(
                failures=failures[name],
                targets={
                    t: summarize(estimates[:, v, k], ses[:, v, k], truth[t])
                    for k, t in enumerate(TARGETS)
                },
            )
            for v, name in enumerate(variants)
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["diagnostic", "qualification"], required=True)
    parser.add_argument("--replications", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.replications < 2 or args.workers < 1:
        parser.error("Require at least two replications and one worker")
    if args.output.resolve() in [
        p.resolve()
        for p in BASELINE.parent.glob("qualification*.json")
        if p.name in ("qualification.json", "qualification-pre-review.json")
    ]:
        parser.error("Frozen baseline records cannot be overwritten")
    baseline = json.loads(BASELINE.read_text())
    cases = diagnostic_cases() if args.mode == "diagnostic" else baseline["cases"]
    sources = [
        "diff_diff/chained_did.py",
        "diff_diff/chained_did_results.py",
        "benchmarks/python/validate_chained_did.py",
        "benchmarks/python/diagnose_chained_did.py",
    ]
    report = dict(
        status="running",
        mode=args.mode,
        replications_per_case=args.replications,
        seed=args.seed,
        seed_derivation="SeedSequence([seed,case_index,replication]).generate_state(1,dtype=uint64)[0]",
        python=platform.python_version(),
        numpy=np.__version__,
        pandas=pd.__version__,
        scipy=scipy.__version__,
        baseline_sha256=hashlib.sha256(BASELINE.read_bytes()).hexdigest(),
        source_hashes={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},
        cases=cases,
        results={},
        acceptance_rule=baseline["thresholds"],
        interpretation="Oracle variants are diagnostics, not public estimator options. Reference agreement is a separate report.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        tmp = args.output.with_suffix(".tmp")
        tmp.write_text(json.dumps(report, indent=2) + "\n")
        tmp.replace(args.output)

    save()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = {
            pool.submit(run_case, c, args.replications, args.seed, i, args.mode == "diagnostic"): i
            for i, c in enumerate(cases)
        }
        for job in as_completed(jobs):
            i = jobs[job]
            report["results"][str(i)] = job.result()
            save()
            print(f"Completed {args.mode} case {i+1}/{len(cases)}", flush=True)
    report["status"] = "complete"
    if args.mode == "qualification":
        report["core_thresholds_pass"] = all(
            t["meets_prespecified_thresholds"]
            for r in report["results"].values()
            if r["design"]["violation"] == "none"
            for t in r["variants"]["fitted_empirical"]["targets"].values()
        )
    save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
