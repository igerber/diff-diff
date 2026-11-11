"""Prespecified native chained-DiD qualification study.

Run from the repository root with PYTHONPATH=. and single-threaded BLAS:
python benchmarks/python/validate_chained_did.py --replications 2000 --workers 4
This runs fits; it does not install or modify any environment. Negative designs
are sensitivity evidence, not calibration acceptance cases.
"""

import argparse
import hashlib
import json
import platform
import subprocess
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm

from diff_diff import ChainedDiD


def generate_data(
    n: int,
    observation: str,
    seed: int,
    *,
    null: bool = False,
    clusters: int = 0,
    violation: str = "none",
) -> tuple:
    """Generate two cohorts with analytic superpopulation ATT targets.

    X is equiprobable binary. Cohort probabilities at each X are proportional
    to (1, exp(-.2+.7X), exp(-.4-.4X)). Untreated changes are linear in X,
    with an unrestricted pre-reference cohort jump and AR(.5) errors. Cohort
    2 effects are (t-1)*(1+.5X), cohort 3 effects (t-2)*(1.5+.5X).
    """
    if observation not in ("balanced", "rotating", "intermittent", "cohort_dependent"):
        raise ValueError("Unknown observation design")
    if violation not in ("none", "selection", "parallel_trends", "propensity"):
        raise ValueError("Unknown assumption violation")
    rng = np.random.default_rng(seed)
    levels = np.array([0.0, 1.0, 2.0]) if violation == "propensity" else np.array([0.0, 1.0])
    x = rng.choice(levels, n)

    def probabilities(values):
        z = values**2 if violation == "propensity" else values
        utilities = np.column_stack(
            [np.ones(len(values)), np.exp(-0.2 + 0.7 * z), np.exp(-0.4 - 0.4 * z)]
        )
        return utilities / utilities.sum(axis=1)[:, None]

    population_prob = probabilities(levels)
    prob = probabilities(x)
    group_index = (rng.random(n)[:, None] > np.cumsum(prob, axis=1)).sum(axis=1)
    cohort = np.array([0, 2, 3])[group_index]
    cohort_mass = population_prob.mean(axis=0)
    x_mean = (population_prob * levels[:, None]).mean(axis=0) / cohort_mass
    truth = {
        "g2_t3": 2 + x_mean[1],
        "g3_t4": 3 + x_mean[2],
        "overall": (cohort_mass[1] * (6 + 3 * x_mean[1]) + cohort_mass[2] * (4.5 + 1.5 * x_mean[2]))
        / (3 * cohort_mass[1] + 2 * cohort_mass[2]),
    }
    if null:
        truth = {k: 0.0 for k in truth}
    t = np.arange(5)
    error = rng.normal(size=(n, 5))
    for j in range(1, 5):
        error[:, j] = 0.5 * error[:, j - 1] + np.sqrt(0.75) * error[:, j]
    cluster_ids = np.arange(n) % clusters if clusters else np.arange(n)
    if clusters:
        error += rng.normal(size=(clusters, 5))[cluster_ids]
    y = rng.normal(scale=2, size=(n, 1)) + 0.5 * t + 0.4 * x[:, None] * t + error
    for g, base_effect in ((2, 1), (3, 1.5)):
        member = cohort == g
        y[member] += 2 * (t >= g - 1)
        if not null:
            y[member] += np.maximum(t - g + 1, 0) * (base_effect + 0.5 * x[member, None])
        if violation == "parallel_trends":
            y[member] += 0.4 * np.maximum(t - (g - 1), 0)
    observed = np.ones((n, 5), dtype=bool)
    if observation in ("rotating", "cohort_dependent"):
        if observation == "rotating":
            window = rng.integers(0, 4, n)
        else:
            rates = np.array([[0.4, 0.3, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4], [0.25, 0.25, 0.25, 0.25]])
            window = (rng.random(n)[:, None] > np.cumsum(rates[group_index], axis=1)).sum(axis=1)
        observed = (t[None, :] == window[:, None]) | (t[None, :] == window[:, None] + 1)
    elif observation == "intermittent":
        observed = rng.random((n, 5)) < 0.7
    if violation == "selection":
        observed &= rng.random((n, 5)) < (0.3 + 0.5 * (x[:, None] > 0))
    data = pd.DataFrame(
        {
            "id": np.repeat(np.arange(n), 5),
            "t": np.tile(t, n),
            "g": np.repeat(cohort, 5),
            "x": np.repeat(x, 5),
            "y": np.where(observed, y, np.nan).ravel(),
            "cluster": np.repeat(cluster_ids, 5),
        }
    )
    return data, truth


def generate_sparse_data(n: int, sparse_treated: int, seed: int, sparse_controls: int = 19):
    """Return an outcome-independent sparse-date panel with ATT(3,3)=1.

    Half the units are never treated, half first treated at date 3. All have
    dates 2 and 3; the first specified units per arm also have date 1. When
    sparse_treated is positive, these extra-date units omit date 4. Unit
    intercepts and independent unit-date errors are standard normal.
    """
    if n < 2 or n % 2 or not 0 <= sparse_treated <= n // 2 or not 0 <= sparse_controls <= n // 2:
        raise ValueError("Sparse design counts require even n and arm counts between 0 and n/2")
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        g = 3 if i % 2 else 0
        intercept = rng.normal()
        sparse = i // 2 < (sparse_treated if g else sparse_controls)
        for t in (1, 2, 3, 4):
            if (t == 1 and not sparse) or (t == 4 and sparse and sparse_treated > 0):
                continue
            y = intercept + t + rng.normal() + float(g > 0 and t >= g)
            rows.append((i, t, g, y))
    return pd.DataFrame(rows, columns=["id", "t", "g", "y"]), {"g3_t3": 1.0}


def summarize(estimates: np.ndarray, ses: np.ndarray, truth: float) -> dict:
    """Report unconditional coverage, failures and Monte Carlo uncertainty."""
    valid = np.isfinite(estimates) & np.isfinite(ses) & (ses > 0)
    r = len(estimates)
    e, s = estimates[valid], ses[valid]
    lower, upper = estimates - norm.ppf(0.975) * ses, estimates + norm.ppf(0.975) * ses
    covered = valid & (lower <= truth) & (upper >= truth)
    rejected = valid & ((lower > 0) | (upper < 0))
    sd = float(np.std(e, ddof=1)) if len(e) > 1 else np.nan
    bias = float(np.mean(e) - truth) if len(e) else np.nan
    ratio = float(np.mean(s * s) / (sd * sd)) if sd > 0 else np.nan
    mcse = float(np.sqrt(0.95 * 0.05 / r))
    coverage = float(covered.mean())
    rejection = float(rejected.mean())
    passed = bool(
        valid.all()
        and abs(bias) < 0.1 * sd
        and 0.9 <= ratio <= 1.1
        and abs(coverage - 0.95) <= 3 * mcse
    )
    if truth == 0:
        passed = passed and abs(rejection - 0.05) <= 3 * mcse
    return {
        "truth": truth,
        "replications": r,
        "failures": int((~valid).sum()),
        "bias": bias,
        "bias_mcse": sd / np.sqrt(len(e)) if len(e) else np.nan,
        "rmse": float(np.sqrt(np.mean((e - truth) ** 2))) if len(e) else np.nan,
        "empirical_sd": sd,
        "mean_variance": float(np.mean(s * s)) if len(s) else np.nan,
        "variance_ratio": ratio,
        "coverage": coverage,
        "coverage_mcse": float(np.sqrt(coverage * (1 - coverage) / r)),
        "nominal_binomial_mcse": mcse,
        "coverage_given_valid": float(covered[valid].mean()) if valid.any() else np.nan,
        "rejection_at_zero": rejection,
        "rejection_mcse": float(np.sqrt(rejection * (1 - rejection) / r)),
        "mean_interval_width": float(2 * norm.ppf(0.975) * np.mean(s)) if len(s) else np.nan,
        "meets_prespecified_thresholds": passed,
    }


def _run_case(case: dict, replications: int, seed: int, case_index: int) -> dict:
    started = time.monotonic()
    sparse = "sparse_treated" in case
    names = ("g3_t3",) if sparse else ("overall", "g2_t3", "g3_t4")
    estimates = np.full((replications, len(names)), np.nan)
    ses = np.full_like(estimates, np.nan)
    pair_diagnostics = dict(
        replications_with_singleton_exclusions=0, replications_with_low_support=0
    )
    failures = {}
    truth = None
    for j in range(replications):
        run_seed = int(np.random.SeedSequence([seed, case_index, j]).generate_state(1)[0])
        if sparse:
            data, truth = generate_sparse_data(
                case["n"], case["sparse_treated"], run_seed, case["sparse_controls"]
            )
        else:
            data, truth = generate_data(seed=run_seed, **case)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="ChainedDiD pair support:", category=UserWarning
                )
                result = ChainedDiD(cluster="cluster" if case.get("clusters") else None).fit(
                    data, "y", "id", "t", "g", covariates=None if sparse else ["x"]
                )
            if sparse:
                cell = result.group_time_effects[3, 3]
                estimates[j], ses[j] = [cell.effect], [cell.se]
            else:
                cells = [result.group_time_effects[2, 3], result.group_time_effects[3, 4]]
                estimates[j] = [result.att, cells[0].effect, cells[1].effect]
                ses[j] = [result.se, cells[0].se, cells[1].se]
            pair_diagnostics["replications_with_singleton_exclusions"] += int(
                any(
                    row["reason"] == "insufficient_pair_replication"
                    for row in result.diagnostics["excluded_links"]
                )
            )
            pair_diagnostics["replications_with_low_support"] += int(
                any(row["low_support"] for row in result.diagnostics["links"])
            )
        except (ValueError, np.linalg.LinAlgError) as exc:
            key = str(exc)
            failures[key] = failures.get(key, 0) + 1
    return {
        "design": case,
        "elapsed_seconds": time.monotonic() - started,
        "failures": failures,
        "pair_diagnostics": pair_diagnostics,
        "targets": {
            name: summarize(estimates[:, k], ses[:, k], truth[name]) for k, name in enumerate(names)
        },
    }


def main() -> int:
    """Run the frozen scenario grid and write an explicitly labelled manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replications", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--design", choices=("full", "sparse"), default="full")
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/data/chained_did/qualification.json")
    )
    args = parser.parse_args()
    if args.replications < 2 or args.workers < 1:
        parser.error("replications must be >=2 and workers >=1")
    frozen = Path(__file__).resolve().parents[1] / "data/chained_did"
    protected = {
        frozen / name
        for name in (
            "qualification.json",
            "qualification-pre-review.json",
            "qualification-followup.json",
            "calibration-diagnostics.json",
            "reference-validation.json",
        )
    }
    if args.output.resolve() in protected or args.output.exists():
        parser.error(
            "Choose a new --output path; existing and frozen evidence cannot be overwritten"
        )
    cases = []
    for n in (1000, 4000):
        for observation in ("balanced", "rotating", "intermittent", "cohort_dependent"):
            cases.append(
                dict(
                    n=n,
                    observation=observation,
                    clusters=(50 if n == 1000 else 100) if observation == "intermittent" else 0,
                    null=False,
                    violation="none",
                )
            )
        cases.append(dict(n=n, observation="rotating", clusters=0, null=True, violation="none"))
    for violation in ("selection", "parallel_trends", "propensity"):
        cases.append(
            dict(n=1000, observation="intermittent", clusters=0, null=False, violation=violation)
        )
    if args.design == "sparse":
        cases = [
            dict(n=400, sparse_treated=k, sparse_controls=19, violation="none")
            for k in (0, 1, 2, 5, 20, 50)
        ]
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    source_commit = revision.stdout.strip() if revision.returncode == 0 else None
    source_uncommitted = None
    if source_commit is not None:
        state = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
        source_uncommitted = bool(state.stdout.strip()) if state.returncode == 0 else None
    manifest = {
        "status": "running",
        "scope": "qualification" if args.replications >= 2000 else "pilot",
        "replications_per_case": args.replications,
        "seed": args.seed,
        "design": args.design,
        "seed_derivation": "SeedSequence([seed, case_index, replication]).generate_state(1)[0]",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "backend": "native Python estimator",
        "source_commit": source_commit,
        "source_uncommitted": source_uncommitted,
        "source_hashes": {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in (
                "diff_diff/chained_did.py",
                "diff_diff/chained_did_results.py",
                "benchmarks/python/validate_chained_did.py",
            )
        },
        "cases": cases,
        "results": {},
        "external_reference": "not run in this invocation; offline regression checks use frozen public R outputs",
        "thresholds": "zero failures; |bias|<.1 empirical SD; variance ratio [.9,1.1]; 95% coverage and null 5% rejection within 3 binomial MCSE",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(args.output)

    save()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_run_case, case, args.replications, args.seed, i): i
            for i, case in enumerate(cases)
        }
        for future in as_completed(futures):
            i = futures[future]
            manifest["results"][str(i)] = future.result()
            save()
            print(f"Completed scenario {i+1}/{len(cases)}: {cases[i]}", flush=True)
    core = [r for r in manifest["results"].values() if r["design"]["violation"] == "none"]
    manifest["core_thresholds_pass"] = all(
        t["meets_prespecified_thresholds"] for r in core for t in r["targets"].values()
    )
    manifest["status"] = "complete"
    save()
    print(f"Complete: {args.output}; core thresholds pass: {manifest['core_thresholds_pass']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
