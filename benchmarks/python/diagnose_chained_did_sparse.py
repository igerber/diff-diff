"""Same-panel Gaussian covariance diagnostic; oracle quantities are benchmark-only."""

import argparse
import hashlib
import json
import platform
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from benchmarks.python.diagnose_chained_did import replication_seed
from benchmarks.python.validate_chained_did import generate_sparse_data, summarize
from diff_diff import ChainedDiD

VARIANTS = ("identity_hc0", "identity_oracle", "design_precision_hc0", "design_precision_oracle")


def cases():
    """Prespecified fixed-count and proportionally growing support cases."""
    return [
        dict(n=400, sparse_treated=k, sparse_controls=19, regime="baseline")
        for k in (0, 1, 2, 5, 20, 50)
    ] + [
        dict(n=4000, sparse_treated=k, sparse_controls=c, regime=regime)
        for k, c, regime in (
            (2, 19, "fixed"),
            (5, 19, "fixed"),
            (20, 190, "growing"),
            (50, 190, "growing"),
        )
    ]


def sparse_system(data):
    """Build exact conditional innovation covariance for the declared DGP only."""
    registry = data.drop_duplicates("id").set_index("id")
    times = sorted(data.t.unique())
    if set(registry.g) != {0, 3} or 2 not in times or 3 not in times:
        raise ValueError("Diagnostic requires cohorts 0/3 and reference/target dates 2/3")
    y = (
        data.pivot(index="id", columns="t", values="y")
        .reindex(index=registry.index, columns=times)
        .to_numpy()
    )
    treated = registry.g.to_numpy() == 3
    columns = [t for t in times if t != 2]
    pairs, rows, coefficients, patterns, metrics = [], [], [], [], []
    for s, t in combinations(range(len(times)), 2):
        observed = np.isfinite(y[:, s]) & np.isfinite(y[:, t])
        nt, nc = np.count_nonzero(observed & treated), np.count_nonzero(observed & ~treated)
        if min(nt, nc) < 2:
            continue
        row = np.array([float(u == times[t]) - float(u == times[s]) for u in columns])
        weight = np.where(treated, 1 / nt, -1 / nc) * observed
        pairs.append((times[s], times[t]))
        rows.append(row)
        coefficients.append(weight)
        patterns.append(observed)
        metrics.append(nt * nc / (nt + nc))
    a = np.asarray(rows).reshape(-1, len(columns))
    contrast = np.array([float(t == 3) for t in columns])
    inverse = np.linalg.pinv(a)
    if not np.allclose(contrast @ inverse @ a, contrast, atol=1e-12, rtol=0):
        raise ValueError("Diagnostic target is unsupported by the replicated pair catalogue")
    metrics = np.array([np.ones(len(pairs)), metrics])
    loadings = np.array(
        [contrast @ np.linalg.pinv(np.sqrt(h)[:, None] * a) * np.sqrt(h) for h in metrics]
    )
    error_loadings = np.zeros((len(registry), len(times), len(pairs)))
    for j, (s, t) in enumerate(pairs):
        error_loadings[:, times.index(s), j] = -coefficients[j]
        error_loadings[:, times.index(t), j] = coefficients[j]
    b = error_loadings.reshape(-1, len(pairs))
    omega = b.T @ b
    return dict(
        units=registry.index,
        times=times,
        treated=treated,
        pairs=pairs,
        incidence=a,
        coefficients=np.array(coefficients).T,
        observed=np.array(patterns).T,
        metrics=metrics,
        loadings=loadings,
        error_loadings=b,
        omega=omega,
        oracle_variances=np.einsum("ij,jk,ik->i", loadings, omega, loadings),
    )


def evaluate(data, system):
    """Evaluate two point estimators and two covariance choices on one panel."""
    times = system["times"]
    y = (
        data.pivot(index="id", columns="t", values="y")
        .reindex(index=system["units"], columns=times)
        .to_numpy()
    )
    d = np.nan_to_num(y).ravel() @ system["error_loadings"]
    q = np.zeros_like(system["coefficients"])
    for j, (s, t) in enumerate(system["pairs"]):
        observed = system["observed"][:, j]
        target, control = observed & system["treated"], observed & ~system["treated"]
        delta = y[:, times.index(t)] - y[:, times.index(s)]
        centre = np.where(system["treated"], delta[target].mean(), delta[control].mean())
        q[observed, j] = system["coefficients"][observed, j] * (delta[observed] - centre[observed])
    points = system["loadings"] @ d
    hc0_variances = np.sum((q @ system["loadings"].T) ** 2, axis=0)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="ChainedDiD pair support:", category=UserWarning)
        native = ChainedDiD().fit(data, "y", "id", "t", "g")
    cell = native.group_time_effects[3, 3]
    point_error = float(abs(points[0] - cell.effect))
    variance_error = float(abs(hc0_variances[0] - cell.se**2))
    if (
        not np.isfinite([point_error, variance_error]).all()
        or point_error > 1e-10
        or variance_error > 1e-10
    ):
        raise ValueError("Independent identity recovery differs from native estimate or covariance")
    return dict(
        estimates=np.repeat(points, 2),
        ses=np.sqrt(
            np.array(
                [
                    hc0_variances[0],
                    system["oracle_variances"][0],
                    hc0_variances[1],
                    system["oracle_variances"][1],
                ]
            )
        ),
        native_discrepancy=point_error,
        native_variance_discrepancy=variance_error,
    )


def run_case(case, replications, seed, index):
    """Replay actual native fits; retain every failure in unconditional metrics."""
    started = time.monotonic()
    estimates = np.full((replications, 4), np.nan)
    ses = np.full_like(estimates, np.nan)
    failures = {}
    point_error = variance_error = 0.0
    system = None
    for j in range(replications):
        data, _ = generate_sparse_data(
            case["n"],
            case["sparse_treated"],
            replication_seed(seed, index, j),
            case["sparse_controls"],
        )
        try:
            if system is None:
                system = sparse_system(data)
            result = evaluate(data, system)
            estimates[j], ses[j] = result["estimates"], result["ses"]
            point_error = max(point_error, result["native_discrepancy"])
            variance_error = max(variance_error, result["native_variance_discrepancy"])
        except (ValueError, np.linalg.LinAlgError) as exc:
            failures[str(exc)] = failures.get(str(exc), 0) + 1
    return dict(
        design=case,
        elapsed_seconds=time.monotonic() - started,
        failures=failures,
        native_max_abs_error=point_error,
        native_variance_max_abs_error=variance_error,
        oracle_variance_ratio_design_to_identity=(
            float(system["oracle_variances"][1] / system["oracle_variances"][0])
            if system is not None
            else None
        ),
        variants={
            name: summarize(estimates[:, k], ses[:, k], 1.0) for k, name in enumerate(VARIANTS)
        },
    )


def main():
    """Write new evidence without Git or R; refuse existing/frozen output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replications", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/data/chained_did/sparse-covariance-diagnostic.json"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    frozen = root / "benchmarks/data/chained_did"
    protected = {
        frozen / name
        for name in (
            "qualification.json",
            "qualification-pre-review.json",
            "qualification-followup.json",
            "calibration-diagnostics.json",
            "reference-validation.json",
            "qualification-sparse.json",
        )
    }
    if args.replications < 2 or args.workers < 1:
        parser.error("replications must be >=2 and workers >=1")
    if args.output.exists() or args.output.resolve() in protected:
        parser.error(
            "Choose a new --output path; existing and frozen evidence cannot be overwritten"
        )
    sources = [
        "diff_diff/chained_did.py",
        "diff_diff/chained_did_results.py",
        "benchmarks/python/validate_chained_did.py",
        "benchmarks/python/diagnose_chained_did.py",
        "benchmarks/python/diagnose_chained_did_sparse.py",
    ]
    manifest = dict(
        status="running",
        scope="conditional Gaussian covariance diagnostic",
        seed=args.seed,
        replications_per_case=args.replications,
        seed_derivation="SeedSequence([seed,case_index,replication]), uint64",
        python=platform.python_version(),
        numpy=np.__version__,
        pandas=pd.__version__,
        scipy=scipy.__version__,
        backend="native Python estimator; single-threaded BLAS requested externally",
        source_hashes={p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in sources},
        cases=cases(),
        results={},
        statistically_qualified=False,
        thresholds="zero failures; |bias|<.1 empirical SD; variance ratio [.9,1.1]; coverage within 3 nominal binomial MCSE",
        interpretation="Oracle covariance is unavailable in empirical applications; no general inference qualification.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(args.output)

    save()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_case, case, args.replications, args.seed, i): i
            for i, case in enumerate(cases())
        }
        for future in as_completed(futures):
            index = futures[future]
            manifest["results"][str(index)] = future.result()
            save()
            print(f"Completed case {index + 1}/{len(cases())}", flush=True)
    manifest["status"] = "complete"
    manifest["variant_thresholds_pass"] = {
        name: all(
            r["variants"][name]["meets_prespecified_thresholds"]
            for r in manifest["results"].values()
        )
        for name in VARIANTS
    }
    save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
