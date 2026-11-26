"""Matched public-output comparison with pinned cdid; never reads its estimator source."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import root
from scipy.special import expit

from benchmarks.python.validate_chained_did import generate_data
from diff_diff import ChainedDiD

IMAGE = "diff-diff-cdid-reference:1.1.0.0"


def validate_environment(expected, actual):
    """Fail before estimation if the selected image differs from its inventory."""

    def inventory(value):
        return (
            value["R"],
            value["platform"],
            sorted((p["Package"], p["Version"], p["Built"]) for p in value["packages"]),
        )

    if inventory(expected) != inventory(actual):
        raise ValueError("Reference image package inventory differs from packages.json")


def compare(native, reference, *, outcome_sd, clusters=None):
    """Compare original labels/registry and predeclared estimator-scale inference."""
    meta = reference["metadata"]
    if meta["covariance_scale"] != "estimator":
        raise ValueError("Reference must declare estimator covariance scale")
    if meta["cluster_correction"] != "none":
        raise ValueError("Reference must declare its raw cluster correction")
    correction = clusters / (clusters - 1) if clusters is not None else 1.0
    rows = reference["cells"]
    keys = [(r["group"], r["time"]) for r in rows]
    checks = {
        "registry": meta["n"] == native.n_units,
        "labels": len(set(keys)) == len(keys) and set(keys) == set(native.vcov_index),
    }
    atol = 1e-7 * max(1.0, outcome_sd)
    cov_atol = 1e-7 * max(1.0, outcome_sd**2)

    def close(a, b, *, covariance=False):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        return bool(
            a.shape == b.shape
            and np.allclose(
                a,
                b,
                rtol=1e-5 if covariance else 1e-6,
                atol=cov_atol if covariance else atol,
                equal_nan=True,
            )
        )

    if checks["labels"]:
        order = [keys.index(k) for k in native.vcov_index]
        checks["support"] = all(
            rows[i]["available"] == (native.group_time_effects[k].skip_reason is None)
            for k, i in zip(native.vcov_index, order)
        )
        checks["estimates"] = close(
            [native.group_time_effects[k].effect for k in native.vcov_index],
            [rows[i]["att"] for i in order],
        )
        covariance = np.asarray(reference["vcov"], dtype=float)[np.ix_(order, order)] * correction
        checks["covariance"] = close(native.vcov, covariance, covariance=True)
        finite = np.isfinite(covariance) & np.isfinite(native.vcov)
        covariance_error = (
            float(np.max(np.abs(covariance[finite] - native.vcov[finite])))
            if finite.any()
            else None
        )
    else:
        checks.update(support=False, estimates=False, covariance=False)
        covariance_error = None
    checks["overall_estimate"] = close(native.att, reference["overall"]["att"])
    ref_se = np.asarray(reference["overall"]["se"], dtype=float) * np.sqrt(correction)
    checks["overall_se"] = close(native.se, ref_se)
    if "events" in reference:
        event = native.aggregate("event_study")
        r = reference["events"]
        event_keys = list(r["event_time"])
        checks["event_labels"] = set(event_keys) == set(event.event_time) and len(
            event_keys
        ) == len(event.event_time)
        if checks["event_labels"]:
            order = [event_keys.index(t) for t in event.event_time]
            checks["event_estimates"] = close(event.att, np.asarray(r["att"], dtype=float)[order])
            rse = np.asarray(r["se"], dtype=float)[order] * np.sqrt(correction)
            refs = np.asarray(r["is_reference"], dtype=bool)[order]
            checks["event_reference_roles"] = bool(np.array_equal(refs, event.is_reference))
            # Canonical Python references carry NaN inference; R carries exact zero SE.
            checks["event_reference_se"] = bool(
                np.all(rse[refs] == 0) and np.isnan(event.se[event.is_reference]).all()
            )
            checks["event_se"] = close(event.se[~event.is_reference], rse[~event.is_reference])
            checks["event_covariance"] = close(
                event.vcov,
                np.asarray(r["vcov"], dtype=float)[np.ix_(order, order)] * correction,
                covariance=True,
            )
    if "events" not in reference:
        checks["event_schema"] = False
    return dict(
        passed=all(checks.values()),
        checks=checks,
        covariance_max_abs_error=covariance_error,
        cluster_cr1_factor_applied=correction,
        tolerances=dict(
            estimate_rtol=1e-6, estimate_atol=atol, covariance_rtol=1e-5, covariance_atol=cov_atol
        ),
    )


def rotating(windows=((0, 1), (1, 2), (2, 3))):
    rows = []
    for cohort in (0, 2):
        for w in windows:
            for noise in (-1.5, -0.5, 0.5, 1.5):
                uid = max([r[0] for r in rows], default=0) + 1
                for t in w:
                    y = 3 * t + noise * t
                    if cohort:
                        y += 10 + 2 * (t >= 1) + {2: 3, 3: 7}.get(t, 0)
                    rows.append((uid, t, cohort, y))
    data = pd.DataFrame(rows, columns=["id", "t", "g", "y"])
    for t in set(range(4)) - set(data.t):
        data = pd.concat([data, data.iloc[[0]].assign(t=t, y=np.nan)], ignore_index=True)
    return data


def fixtures():
    result = {
        "rotating": rotating(),
        "earlier_bridge": rotating(((0, 1), (0, 3))),
        "nonadjacent": rotating(((1, 3),)),
    }
    for name, observation, clusters in (
        ("balanced_binary_x", "balanced", 0),
        ("intermittent_shared_cluster", "intermittent", 40),
    ):
        data, _ = generate_data(800, observation, 110, clusters=clusters)
        result[name] = data if clusters else data.drop(columns="cluster")
    result["two_date"] = rotating(((1, 2),)).query("t >= 1 and t <= 2").copy()
    data = result["intermittent_shared_cluster"]
    extra = [
        dict(id=800 + j, t=0, g=3, x=j % 2, y=np.nan if j < 2 else 0.0, cluster=j) for j in range(4)
    ]
    result["registry_units"] = pd.concat([data, pd.DataFrame(extra)], ignore_index=True)
    rng = np.random.default_rng(141)
    n = 800
    x = rng.normal(size=n)
    g = np.where(rng.random(n) < expit(0.2 + 0.5 * x), 2, 0)
    y = (
        rng.normal(size=(n, 4))
        + np.arange(4) * x[:, None]
        + 3 * ((g[:, None] > 0) & (np.arange(4) >= 2))
    )
    continuous = pd.DataFrame(
        dict(
            id=np.repeat(np.arange(n), 4),
            t=np.tile(np.arange(4), n),
            g=np.repeat(g, 4),
            x=np.repeat(x, 4),
            y=y.ravel(),
        )
    )
    result["balanced_continuous_x"] = continuous
    missing = rng.random((n, 4)) > np.where(g[:, None] > 0, 0.25, 0.1)
    result["incomplete_continuous_x"] = continuous.assign(y=np.where(missing, y, np.nan).ravel())
    for frame in result.values():
        frame["id"] = pd.factorize(frame.id, sort=True)[0] + 1
    return result


def calibrated_points(data):
    """Independent weighted-score diagnostic; returns points, never inference."""
    registry = data.drop_duplicates("id")
    times = sorted(data.t.unique())
    y = (
        data.pivot(index="id", columns="t", values="y")
        .reindex(index=registry.id, columns=times)
        .to_numpy()
    )
    membership, x = registry.g.to_numpy(), registry.x.to_numpy()
    eligible = np.isfinite(y).sum(axis=1) >= 2
    points = {}
    for g in sorted(set(membership) - {0}):
        base = times.index(g) - 1
        cols = [j for j in range(len(times)) if j != base]
        rows, values, direct = [], [], {}
        for s in range(len(times)):
            for t in range(s + 1, len(times)):
                observed = np.isfinite(y[:, s]) & np.isfinite(y[:, t])
                selected = observed & ((membership == g) | (membership == 0))
                treated = membership[selected] == g
                if not treated.any() or treated.all():
                    continue
                z = np.column_stack([np.ones(selected.sum()), x[selected]])
                rate_t = treated.sum() / ((membership == g) & eligible).sum()
                rate_c = (~treated).sum() / ((membership == 0) & eligible).sum()
                calibration = np.where(treated, 1 / rate_t, 1 / rate_c)
                fitted = root(
                    lambda b: z.T @ (calibration * (treated - expit(z @ b))), np.zeros(z.shape[1])
                )
                if not fitted.success:
                    raise ValueError("Calibrated-score diagnostic did not converge")
                odds = np.exp(z @ fitted.x)
                delta = y[selected, t] - y[selected, s]
                w = (~treated) * odds
                contrast = delta[treated].mean() - w @ delta / w.sum()
                row = np.zeros(len(times))
                row[s], row[t] = -1, 1
                rows.append(row[cols])
                values.append(contrast)
                direct[s, t] = contrast
        a = np.asarray(rows)
        if np.linalg.matrix_rank(a) != len(cols):
            raise ValueError("Calibrated-score diagnostic requires full rank")
        recovered = np.linalg.lstsq(a, values, rcond=None)[0]
        for t, label in enumerate(times):
            points[g, label] = (
                0.0
                if t == base
                else (-direct[t, base] if t < base else float(recovered[cols.index(t)]))
            )
    return points


def finite_json(value):
    """Represent undefined inference as JSON null in external-reference records."""
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=IMAGE)
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/data/chained_did/reference-validation.json")
    )
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    args.workspace.mkdir(parents=True, exist_ok=True)
    work = args.workspace.resolve()
    script = root / "benchmarks/R/benchmark_chained_did.R"
    probe = root / "benchmarks/R/inspect_chained_did_weights.R"
    environment = root / "benchmarks/environments/chained_did/packages.json"
    installed = json.loads(
        subprocess.check_output(
            [
                "docker",
                "run",
                "--rm",
                args.image,
                "Rscript",
                "-e",
                "jsonlite::write_json(list(R=R.version.string, platform=R.version$platform, "
                'packages=as.data.frame(installed.packages()[,c("Package","Version","Built")])), '
                "stdout(), auto_unbox=TRUE)",
            ],
            text=True,
        )
    )
    validate_environment(json.loads(environment.read_text()), installed)
    report = dict(
        status="running",
        image=args.image,
        image_id=subprocess.check_output(
            ["docker", "image", "inspect", args.image, "--format", "{{.Id}}"], text=True
        ).strip(),
        environment_sha256=hashlib.sha256(environment.read_bytes()).hexdigest(),
        environment_verified=True,
        installed_environment=installed,
        source_hashes={
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                script,
                probe,
                root / "diff_diff/chained_did.py",
                root / "diff_diff/chained_did_results.py",
                Path(__file__).resolve(),
            ]
        },
        cases={},
    )
    for name, data in fixtures().items():
        data.to_csv(work / f"{name}.csv", index=False)
        native = ChainedDiD(cluster="cluster" if "cluster" in data else None).fit(
            data, "y", "id", "t", "g", covariates=["x"] if "x" in data else None
        )
        command = [
            "docker",
            "run",
            "--rm",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "-v",
            f"{root}:/repo:ro",
            "-v",
            f"{work}:/work",
            args.image,
            "Rscript",
            "/repo/benchmarks/R/benchmark_chained_did.R",
            f"/work/{name}.csv",
            f"/work/{name}.json",
        ]
        run = subprocess.run(command, capture_output=True, text=True)
        if run.returncode:
            report["cases"][name] = dict(
                passed=False,
                error=run.stderr[-6000:],
                input_sha256=hashlib.sha256((work / f"{name}.csv").read_bytes()).hexdigest(),
            )
        else:
            reference = json.loads((work / f"{name}.json").read_text())
            k = data.drop_duplicates("id").cluster.nunique() if "cluster" in data else None
            outcome = compare(native, reference, outcome_sd=float(data.y.std()), clusters=k)
            if not outcome["checks"].get("estimates", False) and "x" in data:
                diagnostic = calibrated_points(data)
                expected = [row["att"] for row in reference["cells"]]
                actual = [diagnostic[row["group"], row["time"]] for row in reference["cells"]]
                outcome["calibrated_score_point_diagnostic"] = dict(
                    matches_reference=bool(
                        np.allclose(
                            actual,
                            np.asarray(expected, dtype=float),
                            rtol=1e-6,
                            atol=1e-7 * max(1.0, float(data.y.std())),
                            equal_nan=True,
                        )
                    ),
                    points=[dict(group=g, time=t, att=v) for (g, t), v in diagnostic.items()],
                    interpretation="Weighted-score point diagnostic only; native point criterion and inference remain unchanged.",
                )
            outcome["reference"] = reference
            outcome["native"] = native.to_dict()
            report["cases"][name] = outcome
        report["cases"][name]["input_sha256"] = hashlib.sha256(
            (work / f"{name}.csv").read_bytes()
        ).hexdigest()
        print(f"Reference {name}: {report['cases'][name]['passed']}", flush=True)
    probe_command = command[:-3] + [
        "/repo/benchmarks/R/inspect_chained_did_weights.R",
        "/work/incomplete_continuous_x.csv",
        "/work/public-input-probe.json",
    ]
    subprocess.run(probe_command, check=True, capture_output=True, text=True)
    report["public_input_probe"] = json.loads((work / "public-input-probe.json").read_text())
    report["public_input_probe"]["input_sha256"] = report["cases"]["incomplete_continuous_x"][
        "input_sha256"
    ]
    report["status"] = "complete"
    report["all_comparisons_pass"] = all(r["passed"] for r in report["cases"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(finite_json(report), indent=2, allow_nan=False) + "\n")
    return 0 if report["all_comparisons_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
