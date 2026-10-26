"""
Validate that optimization changes produce identical results.

Usage:
    # Save baseline (run BEFORE code changes):
    python benchmarks/speed_review/validate_results.py --save

    # Validate (run AFTER code changes):
    python benchmarks/speed_review/validate_results.py --check
"""

import argparse
import json
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from diff_diff import CallawaySantAnna


def generate_data(n_units=10_000, seed=42, n_covariates=0):
    """Generate deterministic test data."""
    rng = np.random.default_rng(seed)
    n_periods = 10
    treatment_periods = [3, 5, 7]

    cohort_assignment = rng.choice(
        [0] + treatment_periods,
        size=n_units,
        p=[0.25, 0.25, 0.25, 0.25],
    )

    rows = []
    for i in range(n_units):
        g = cohort_assignment[i]
        for t in range(1, n_periods + 1):
            treated = 1 if (g > 0 and t >= g) else 0
            y = rng.normal(0, 1) + 2.0 * treated
            rows.append((i, t, y, g))

    df = pd.DataFrame(rows, columns=["unit", "time", "outcome", "first_treat"])

    if n_covariates > 0:
        cov_rng = np.random.default_rng(seed + 1)
        for i in range(1, n_covariates + 1):
            df[f"x{i}"] = cov_rng.normal(size=len(df))

    return df


def run_estimator(df, estimation_method="reg", covariates=None, control_group="never_treated"):
    """Run estimator and extract key results."""
    cs = CallawaySantAnna(
        n_bootstrap=199,
        seed=42,
        estimation_method=estimation_method,
        control_group=control_group,
    )
    results = cs.fit(
        df,
        outcome="outcome",
        unit="unit",
        time="time",
        first_treat="first_treat",
        covariates=covariates,
    )
    # Post-fit containers (the 3.x fit(aggregate="all") route, removed at 4.0);
    # the fit is bootstrapped, so each call replays the fit-time bootstrap.
    es_container = results.aggregate("event_study")
    grp_container = results.aggregate("group")

    out = {
        "overall_att": float(results.att),
        "overall_se": float(results.se),
        "overall_p_value": float(results.p_value),
        "overall_ci": [float(results.conf_int[0]), float(results.conf_int[1])],
    }

    # Group-time effects (sorted for determinism)
    gt_effects = {}
    for (g, t), data in sorted(results.group_time_effects.items()):
        key = f"{g},{t}"
        gt_effects[key] = {
            "effect": float(data["effect"]),
            "se": float(data["se"]),
            "t_stat": float(data["t_stat"]),
            "p_value": float(data["p_value"]),
            "conf_int": [float(data["conf_int"][0]), float(data["conf_int"][1])],
        }
    out["group_time_effects"] = gt_effects

    # Event study (same keyed shape as the legacy dict, one record per
    # container row including the reference row)
    es = {}
    for row in es_container.to_dataframe().sort_values("event_time").itertuples(index=False):
        es[str(int(row.event_time))] = {
            "effect": float(row.att),
            "se": float(row.se),
            "t_stat": float(row.t_stat),
            "p_value": float(row.p_value),
            "conf_int": [float(row.conf_int_lower), float(row.conf_int_upper)],
        }
    out["event_study"] = es

    # Group effects
    ge = {}
    for row in grp_container.to_dataframe().sort_values("label").itertuples(index=False):
        ge[str(row.label)] = {
            "effect": float(row.att),
            "se": float(row.se),
            "t_stat": float(row.t_stat),
            "p_value": float(row.p_value),
            "conf_int": [float(row.conf_int_lower), float(row.conf_int_upper)],
        }
    out["group_effects"] = ge

    return out


SCENARIOS = [
    {"name": "reg_nocov", "method": "reg", "n_cov": 0},
    {"name": "reg_2cov", "method": "reg", "n_cov": 2},
    {"name": "reg_10cov", "method": "reg", "n_cov": 10},
    {"name": "dr_2cov", "method": "dr", "n_cov": 2},
    {"name": "ipw_2cov", "method": "ipw", "n_cov": 2},
    {"name": "ipw_2cov_nyt", "method": "ipw", "n_cov": 2, "control_group": "not_yet_treated"},
    {"name": "dr_2cov_nyt", "method": "dr", "n_cov": 2, "control_group": "not_yet_treated"},
    {"name": "reg_2cov_nyt", "method": "reg", "n_cov": 2, "control_group": "not_yet_treated"},
]


def save_baseline(path="benchmarks/speed_review/baseline_results.json"):
    """Save baseline results for all scenarios."""
    all_results = {}
    for scenario in SCENARIOS:
        name = scenario["name"]
        print(f"Running scenario: {name} ...")
        df = generate_data(n_covariates=scenario["n_cov"])
        covariates = (
            [f"x{i}" for i in range(1, scenario["n_cov"] + 1)] if scenario["n_cov"] > 0 else None
        )
        control_group = scenario.get("control_group", "never_treated")
        results = run_estimator(
            df,
            estimation_method=scenario["method"],
            covariates=covariates,
            control_group=control_group,
        )
        all_results[name] = results
        print(f"  Overall ATT: {results['overall_att']:.10f}")
        print(f"  N group-time effects: {len(results['group_time_effects'])}")

    with open(path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nBaseline saved to {path}")


def check_results(path="benchmarks/speed_review/baseline_results.json", tol=1e-12):
    """Check current results against baseline for all scenarios."""
    with open(path) as f:
        all_baseline = json.load(f)

    all_failures = []

    for scenario in SCENARIOS:
        name = scenario["name"]
        if name not in all_baseline:
            print(f"  Skipping {name} (no baseline)")
            continue

        baseline = all_baseline[name]
        df = generate_data(n_covariates=scenario["n_cov"])
        covariates = (
            [f"x{i}" for i in range(1, scenario["n_cov"] + 1)] if scenario["n_cov"] > 0 else None
        )
        control_group = scenario.get("control_group", "never_treated")

        # Use relaxed tolerance for covariate scenarios (Cholesky vs lstsq)
        scenario_tol = 1e-10 if scenario["n_cov"] > 0 else tol

        current = run_estimator(
            df,
            estimation_method=scenario["method"],
            covariates=covariates,
            control_group=control_group,
        )

        failures = []

        def compare(label, base_val, cur_val, t):
            if np.isnan(base_val) and np.isnan(cur_val):
                return
            if np.isnan(base_val) or np.isnan(cur_val):
                failures.append(f"  {label}: NaN mismatch baseline={base_val}, current={cur_val}")
                return
            diff = abs(base_val - cur_val)
            if diff > t:
                failures.append(
                    f"  {label}: baseline={base_val:.15e}, current={cur_val:.15e}, diff={diff:.2e}"
                )

        compare(
            f"{name}/overall_att", baseline["overall_att"], current["overall_att"], scenario_tol
        )
        compare(f"{name}/overall_se", baseline["overall_se"], current["overall_se"], scenario_tol)
        compare(
            f"{name}/overall_p_value", baseline["overall_p_value"], current["overall_p_value"], 0.02
        )

        # Compare overall CI values
        if "overall_ci" in baseline and "overall_ci" in current:
            for i, label in enumerate(["lower", "upper"]):
                compare(
                    f"{name}/overall_ci.{label}",
                    baseline["overall_ci"][i],
                    current["overall_ci"][i],
                    scenario_tol,
                )

        # Group-time SE tolerance: tight for covariate scenarios, relaxed for bootstrap
        gt_se_tol = 1e-8 if scenario["n_cov"] > 0 else 0.01

        for key in baseline["group_time_effects"]:
            b = baseline["group_time_effects"][key]
            c = current["group_time_effects"].get(key, {})
            if not c:
                failures.append(f"  {name}/Missing group-time effect: {key}")
                continue
            compare(f"{name}/gt[{key}].effect", b["effect"], c["effect"], scenario_tol)
            compare(f"{name}/gt[{key}].se", b["se"], c["se"], gt_se_tol)
            if "t_stat" in b and "t_stat" in c:
                compare(f"{name}/gt[{key}].t_stat", b["t_stat"], c["t_stat"], gt_se_tol)
            if "p_value" in b and "p_value" in c:
                compare(f"{name}/gt[{key}].p_value", b["p_value"], c["p_value"], 0.02)
            if "conf_int" in b and "conf_int" in c:
                for i, label in enumerate(["lower", "upper"]):
                    compare(
                        f"{name}/gt[{key}].ci.{label}",
                        b["conf_int"][i],
                        c["conf_int"][i],
                        gt_se_tol,
                    )

        # Compare event study effects/SEs if present
        if "event_study" in baseline and "event_study" in current:
            for e_key in baseline["event_study"]:
                b_es = baseline["event_study"][e_key]
                c_es = current["event_study"].get(e_key, {})
                if not c_es:
                    failures.append(f"  {name}/Missing event study effect: e={e_key}")
                    continue
                compare(f"{name}/es[{e_key}].effect", b_es["effect"], c_es["effect"], scenario_tol)
                compare(f"{name}/es[{e_key}].se", b_es["se"], c_es["se"], gt_se_tol)
                if "t_stat" in b_es and "t_stat" in c_es:
                    compare(f"{name}/es[{e_key}].t_stat", b_es["t_stat"], c_es["t_stat"], gt_se_tol)
                if "p_value" in b_es and "p_value" in c_es:
                    compare(f"{name}/es[{e_key}].p_value", b_es["p_value"], c_es["p_value"], 0.02)
                if "conf_int" in b_es and "conf_int" in c_es:
                    for i, label in enumerate(["lower", "upper"]):
                        compare(
                            f"{name}/es[{e_key}].ci.{label}",
                            b_es["conf_int"][i],
                            c_es["conf_int"][i],
                            gt_se_tol,
                        )

        # Compare group effects if present
        if "group_effects" in baseline and "group_effects" in current:
            for g_key in baseline["group_effects"]:
                b_ge = baseline["group_effects"][g_key]
                c_ge = current["group_effects"].get(g_key, {})
                if not c_ge:
                    failures.append(f"  {name}/Missing group effect: g={g_key}")
                    continue
                compare(f"{name}/ge[{g_key}].effect", b_ge["effect"], c_ge["effect"], scenario_tol)
                compare(f"{name}/ge[{g_key}].se", b_ge["se"], c_ge["se"], gt_se_tol)
                if "t_stat" in b_ge and "t_stat" in c_ge:
                    compare(f"{name}/ge[{g_key}].t_stat", b_ge["t_stat"], c_ge["t_stat"], gt_se_tol)
                if "p_value" in b_ge and "p_value" in c_ge:
                    compare(f"{name}/ge[{g_key}].p_value", b_ge["p_value"], c_ge["p_value"], 0.02)
                if "conf_int" in b_ge and "conf_int" in c_ge:
                    for i, label in enumerate(["lower", "upper"]):
                        compare(
                            f"{name}/ge[{g_key}].ci.{label}",
                            b_ge["conf_int"][i],
                            c_ge["conf_int"][i],
                            gt_se_tol,
                        )

        if failures:
            all_failures.extend(failures)
            print(f"  {name}: FAILED ({len(failures)} mismatches)")
        else:
            print(f"  {name}: PASSED ({len(current['group_time_effects'])} effects checked)")

    if all_failures:
        print("\nVALIDATION FAILED:")
        for f in all_failures:
            print(f)
        sys.exit(1)
    else:
        print("\nALL SCENARIOS PASSED")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true", help="Save baseline results")
    parser.add_argument("--check", action="store_true", help="Check against baseline")
    parser.add_argument("--tol", type=float, default=1e-12, help="Tolerance for comparison")
    args = parser.parse_args()

    if args.save:
        save_baseline()
    elif args.check:
        check_results(tol=args.tol)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
