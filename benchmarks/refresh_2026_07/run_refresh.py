#!/usr/bin/env python3
"""
Headline orchestrator for the 2026-07 public benchmark refresh.

Re-runs the published docs/benchmarks.rst comparisons - BasicDiD/TWFE vs
fixest, MultiPeriodDiD vs fixest, CallawaySantAnna vs R did, SyntheticDiD vs
R synthdid, and the MPDTA real-data validation - with a fair protocol:

- diff-diff arm = the RELEASED wheel (pip install diff-diff==<pin>) in an
  isolated uv venv, never the dev tree (provenance hard-fail).
- R arm at package defaults (fixest/data.table threads untouched), latest
  CRAN versions recorded in metadata.
- Untimed in-process warm-up on BOTH sides; fresh subprocess per replication;
  strictly sequential; median of counted reps published; CV > 10% flagged.
- SDID bootstrap parity: Python placebo n_bootstrap=200 == R vcov placebo
  default 200 (the old orchestrator passed 50 - unfair to R).

Usage:
    python run_refresh.py --setup            # create venvs + preflight only
    python run_refresh.py --smoke            # small-scale pipeline validation
    python run_refresh.py                    # FULL timed run (idle machine!)
    python run_refresh.py --only synthdid    # one estimator
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import refresh_common as rc

sys.path.insert(0, str(rc.REPO_ROOT))

from benchmarks.compare_results import compare_estimates  # noqa: E402
from benchmarks.run_benchmarks import (  # noqa: E402
    SCALE_CONFIGS,
    TIMEOUT_CONFIGS,
)

HEADLINE_PIN = "3.7.0"
HEADLINE_VENV = "dd370"

RESULTS_PATH = rc.RESULTS_DIR / "refresh_results.json"
SMOKE_RESULTS_PATH = rc.RESULTS_DIR / "refresh_results_smoke.json"

# Published-page scope (user-locked 2026-07-10): existing tables only.
BENCH_SPECS: Dict[str, Dict[str, Any]] = {
    # The legacy harness labeled this cell "BasicDiD/TWFE" while both arms
    # actually ran the simple interaction OLS (--type twfe is accepted but
    # changes neither model). The refresh splits it honestly: "basic" is the
    # 2x2 interaction regression, "twfe" is the genuine absorbed-FE pair.
    "basic": {
        "display": "BasicDiD",
        "py_script": "benchmark_basic.py",
        "r_script": "benchmark_fixest.R",
        "dataset": "basic",
        "scales": ["small", "1k", "5k", "10k", "20k"],
        "py_extra": ["--type", "basic"],
        "r_extra": ["--type", "basic"],
        "py_estimator": "diff_diff.DifferenceInDifferences",
        "r_estimator": "fixest::feols",
        "se_rtol": 0.01,
        "se_gate_rtol": 0.01,
    },
    "twfe": {
        "display": "TWFE (absorbed FE)",
        "py_script": "benchmark_twfe.py",
        "r_script": "benchmark_twfe.R",
        "dataset": "basic",
        "scales": ["small", "1k", "5k", "10k", "20k"],
        "py_estimator": "diff_diff.TwoWayFixedEffects",
        "r_estimator": "fixest::feols (absorbed FE)",
        "se_rtol": 0.01,
        "se_gate_rtol": 0.01,
    },
    "multiperiod": {
        "display": "MultiPeriodDiD",
        "py_script": "benchmark_multiperiod.py",
        "r_script": "benchmark_multiperiod.R",
        "dataset": "multiperiod",
        # The published page carries MultiPeriodDiD at small scale only.
        "scales": ["small"],
        "py_estimator": "diff_diff.MultiPeriodDiD",
        "r_estimator": "fixest::feols (multiperiod)",
        "se_rtol": 0.01,
        "se_gate_rtol": 0.01,
    },
    "callaway": {
        "display": "CallawaySantAnna",
        "py_script": "benchmark_callaway.py",
        "r_script": "benchmark_did.R",
        "dataset": "staggered",
        "scales": ["small", "1k", "5k", "10k", "20k"],
        "py_estimator": "diff_diff.CallawaySantAnna",
        "r_estimator": "did::att_gt",
        "se_rtol": 0.10,
        "se_gate_rtol": 0.10,
    },
    "mpdta": {
        "display": "CallawaySantAnna (MPDTA real data)",
        "py_script": "benchmark_callaway.py",
        "r_script": "benchmark_did.R",
        "dataset": "mpdta",
        "scales": ["real"],
        "py_estimator": "diff_diff.CallawaySantAnna",
        "r_estimator": "did::att_gt",
        "se_rtol": 0.02,
        "se_gate_rtol": 0.02,
    },
    # Slowest last: SDID R cells dominate wall-clock.
    "synthdid": {
        "display": "SyntheticDiD",
        "py_script": "benchmark_synthdid.py",
        "r_script": "benchmark_synthdid.R",
        "dataset": "sdid",
        "scales": ["small", "1k", "5k"],
        # Placebo-bootstrap parity with R's vcov(method="placebo") default.
        "py_extra": ["--n-bootstrap", "200"],
        # Jackknife is excluded from total_seconds anyway; skip the wasted
        # wall-clock at scale.
        "r_extra": ["--skip-jackknife", "true"],
        "py_estimator": "diff_diff.SyntheticDiD",
        "r_estimator": "synthdid::synthdid_estimate",
        "se_rtol": 0.10,
        # SDID SE gate is wider than the reporting rtol: both sides estimate
        # the placebo variance by Monte Carlo and R's placebo draw is
        # UNSEEDED, so its SE varies run to run (rep-to-rep values recorded
        # in se_values for audit). 0.35 bounds MC dispersion at the small-N0
        # scale; ATT remains gated at 1e-8 (deterministic Frank-Wolfe).
        "se_gate_rtol": 0.35,
        "slow_scales": {"1k", "5k"},
    },
}

ARMS = ("python_pure", "python_rust", "r")

# Detailed parity surfaces the docs claim match R beyond the headline ATT/SE
# (period effects, group-time effects, event study, group aggregation). Each
# is aligned on its join keys and hard-gated; compact metrics land in the
# committed results JSON under "detail_comparison".
DETAIL_SURFACES: Dict[str, List] = {
    "multiperiod": [("period_effects", ["period"])],
    "callaway": [
        ("group_time_effects", ["group", "time"]),
        ("event_study", ["event_time"]),
        ("group_effects", ["group"]),
    ],
    "mpdta": [
        ("group_time_effects", ["group", "time"]),
        ("event_study", ["event_time"]),
        ("group_effects", ["group"]),
    ],
}


def cell_timeouts(estimator: str, scale: str) -> Dict[str, int]:
    key = "small" if scale == "real" else scale
    t = dict(TIMEOUT_CONFIGS.get(key, TIMEOUT_CONFIGS["small"]))
    if estimator == "synthdid":
        # Warm-up doubles per-process work; R 5k budget 2x -> 7200s.
        t["r"] = t["r"] * 2
        t["python"] = t["python"] * 2
    return t


def multiperiod_extra(scale: str) -> List[str]:
    cfg = SCALE_CONFIGS[scale]["multiperiod"]
    return ["--n-pre", str(cfg["n_pre"]), "--n-post", str(cfg["n_post"])]


def dataset_for(estimator: str, scale: str) -> Path:
    spec = BENCH_SPECS[estimator]
    if spec["dataset"] == "mpdta":
        return rc.ensure_mpdta()
    cfg = SCALE_CONFIGS[scale][spec["dataset"]]
    return rc.ensure_dataset(spec["dataset"], scale, cfg)


def run_cell(
    estimator: str,
    scale: str,
    n_reps: int,
    smoke: bool,
) -> Dict[str, Any]:
    spec = BENCH_SPECS[estimator]
    data_path = dataset_for(estimator, scale)
    timeouts = cell_timeouts(estimator, scale)
    py_extra = list(spec.get("py_extra", []))
    r_extra = list(spec.get("r_extra", []))
    if estimator == "multiperiod":
        py_extra += multiperiod_extra(scale)
        r_extra += multiperiod_extra(scale)

    reps = n_reps
    if scale in spec.get("slow_scales", set()) and not smoke:
        reps = min(n_reps, 4)

    cell: Dict[str, Any] = {
        "estimator": spec["display"],
        "scale": scale,
        "data_file": data_path.name,
        "data_sha256": rc.sha256_file(data_path),
        "n_reps_requested": reps,
        "captured_at_utc": rc.utc_now(),
        "loadavg_at_cell": list(os.getloadavg()),
        "flags": [],
    }
    print(f"\n=== {spec['display']} @ {scale} " f"(reps={reps}, data={data_path.name}) ===")

    arm_results: Dict[str, Dict[str, Any]] = {}
    for arm in ARMS:
        raw_out = rc.RAW_DIR / f"{estimator}_{scale}_{arm}.json"
        if arm == "r":

            def _r_rep(raw_out=raw_out):
                return rc.run_r_rep(
                    spec["r_script"],
                    data_path,
                    raw_out,
                    extra_args=r_extra,
                    timeout=timeouts["r"],
                )

            rep_fn = _r_rep
        else:
            backend = "python" if arm == "python_pure" else "rust"

            def _py_rep(backend=backend, raw_out=raw_out):
                res = rc.run_python_rep(
                    spec["py_script"],
                    data_path,
                    raw_out,
                    backend=backend,
                    venv=HEADLINE_VENV,
                    extra_args=py_extra,
                    timeout=timeouts["python"],
                )
                rc.validate_python_provenance(res, HEADLINE_PIN, HEADLINE_VENV, backend)
                return res

            rep_fn = _py_rep

        arm_out = rc.run_arm(
            f"{estimator}/{scale}/{arm}",
            rep_fn,
            reps,
            allow_cv_rerun=not smoke,
        )
        rc.validate_estimator_field(
            arm_out["result"],
            spec["r_estimator"] if arm == "r" else spec["py_estimator"],
        )
        # compare_estimates reads timing from result["timing"]["stats"].
        arm_out["result"]["timing"] = arm_out["timing"]
        arm_results[arm] = arm_out
        cell["flags"].extend(f"{arm}:{fl}" for fl in arm_out["flags"])
        rc.cooldown(1 if smoke else rc.COOLDOWN_SECONDS)

    # Parity: rust arm vs R (matching the historical "python" arm choice).
    comparison = compare_estimates(
        arm_results["python_rust"]["result"],
        arm_results["r"]["result"],
        spec["display"],
        se_rtol=spec["se_rtol"],
        scale=scale,
        python_pure_results=arm_results["python_pure"]["result"],
        python_rust_results=arm_results["python_rust"]["result"],
    )
    print(
        f"  parity: {'PASS' if comparison.passed else 'FAIL'} "
        f"(ATT diff {comparison.att_diff:.2e}, "
        f"SE rel {comparison.se_rel_diff:.2%})"
    )
    if not comparison.passed:
        cell["flags"].append("parity_fail")
    # CI overlap (published criterion) is gated PER RENDERED ARM inside
    # headline_gate_flags() below - both python_pure and python_rust.

    # Detailed parity surfaces (docs claim per-period / per-(g,t) /
    # event-time / group effects match R - gate them, not just the headline).
    detail: Dict[str, Any] = {}
    for surface, join_keys in DETAIL_SURFACES.get(estimator, []):
        for py_arm in ("python_pure", "python_rust"):
            py_rows = arm_results[py_arm]["result"].get(surface)
            r_rows = arm_results["r"]["result"].get(surface)
            metrics = rc.compare_effect_arrays(py_rows, r_rows, join_keys, se_rtol=spec["se_rtol"])
            detail[f"{surface}:{py_arm}"] = metrics
            tag = f"{surface}:{py_arm}"
            if not metrics["keys_match"]:
                cell["flags"].append(f"detail_keys_mismatch:{tag}")
            if not metrics["att_ok"]:
                cell["flags"].append(f"detail_att_gate_fail:{tag}:{metrics['max_att_diff']}")
            if not metrics["se_ok"]:
                cell["flags"].append(f"detail_se_gate_fail:{tag}:{metrics['max_se_rel_diff']}")
    if detail:
        cell["detail_comparison"] = detail
        n_bad = sum(1 for fl in cell["flags"] if fl.startswith("detail_"))
        print(
            f"  detail surfaces: {len(detail)} compared, "
            f"{'ALL OK' if n_bad == 0 else f'{n_bad} GATE FAILURES'}"
        )

    # Strict headline gates: BOTH rendered Python arms must independently
    # satisfy the documented ATT/SE tolerances vs R. Unlike
    # compare_estimates(), CI overlap is NOT an escape hatch here.
    def _vals(arm: str):
        res = arm_results[arm]["result"]
        return (
            res.get("overall_att", res.get("att")),
            res.get("overall_se", res.get("se")),
        )

    r_att_v, r_se_v = _vals("r")
    for py_arm in ("python_pure", "python_rust"):
        py_att_v, py_se_v = _vals(py_arm)
        cell["flags"].extend(
            rc.headline_gate_flags(
                py_arm,
                py_att_v,
                py_se_v,
                r_att_v,
                r_se_v,
                att_atol=1e-4,
                se_rtol=spec["se_gate_rtol"],
            )
        )

    # Pure-vs-rust backend agreement on the point estimate (hard gate).
    def _att(arm: str) -> float:
        res = arm_results[arm]["result"]
        return float(res.get("overall_att", res.get("att")))

    pure_rust_gap = abs(_att("python_pure") - _att("python_rust"))
    if pure_rust_gap > 1e-8:
        cell["flags"].append(f"pure_rust_att_gate_fail:{pure_rust_gap:.2e}")

    # Known-answer gates.
    if estimator == "mpdta":
        for arm in ARMS:
            res = arm_results[arm]["result"]
            att = float(res.get("overall_att", res.get("att")))
            if abs(att - rc.MPDTA_KNOWN_ATT) > 1e-5:
                cell["flags"].append(f"mpdta_known_answer_fail:{arm}:{att:.6f}")
    if estimator == "synthdid":
        gap = abs(_att("python_rust") - _att("r"))
        if gap > 1e-8:
            # Same Frank-Wolfe algorithm on both sides - this is exactly the
            # check that catches the invalidated-table class of bug.
            cell["flags"].append(f"sdid_att_gate_fail:{gap:.2e}")
        # Weight-vector gates with auditable committed metrics (raw vectors
        # are slimmed out). Both omega and lambda reproduce R at machine
        # precision once aligned by unit/period id (Python's accessor sorts
        # by descending weight; R emits panel order - the id-keyed
        # comparison makes ordering irrelevant). Tight 1e-8 gates on both.
        weight_detail: Dict[str, Any] = cell.setdefault("detail_comparison", {})
        r_res = arm_results["r"]["result"]
        for py_arm in ("python_pure", "python_rust"):
            py_res = arm_results[py_arm]["result"]
            for surface in ("unit_weights", "time_weights"):
                id_field = surface[:-1] + "_ids"
                metrics = rc.compare_weight_vectors(
                    py_res.get(surface),
                    r_res.get(surface),
                    atol=1e-8,
                    py_ids=py_res.get(id_field),
                    r_ids=r_res.get(id_field),
                    # Documented contract: the publication gate is id-aligned
                    # and may never silently degrade to positional order.
                    require_ids=True,
                )
                weight_detail[f"{surface}:{py_arm}"] = metrics
                if not metrics["ok"]:
                    cell["flags"].append(
                        f"sdid_weights_gate_fail:{surface}:{py_arm}:" f"{metrics['max_abs_diff']}"
                    )

    for arm in ARMS:
        entry = dict(arm_results[arm])
        entry["result"] = rc.slim_result(entry["result"])
        cell[arm] = entry
    cell["comparison"] = asdict(comparison)
    return cell


def collect_hard_failures(payload: Dict[str, Any]) -> List[str]:
    """
    Hard failures across EVERY cell in the payload - including stale cells
    preserved by merge-on-write during --only reruns. The runner's exit
    status must reflect the artifact that would be published, not merely
    the cells executed in this invocation.
    """
    failures = []
    for key in sorted(payload.get("cells", {})):
        hard = rc.hard_flags(payload["cells"][key].get("flags", []))
        if hard:
            failures.append(f"{key}: {hard}")
    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--setup", action="store_true", help="Create/refresh venvs + preflight, then exit"
    )
    ap.add_argument("--smoke", action="store_true", help="Small-scale pipeline validation (reps=2)")
    ap.add_argument(
        "--only",
        action="append",
        default=None,
        metavar="ESTIMATOR",
        help="Limit to one estimator (repeatable): " f"{', '.join(BENCH_SPECS)}",
    )
    ap.add_argument(
        "--reps",
        type=int,
        default=8,
        help="Subprocess reps per fast arm (first is excluded "
        "from stats; slow SDID cells cap at 4)",
    )
    ap.add_argument("--force", action="store_true", help="Override the load-average guard")
    args = ap.parse_args()

    rc.RAW_DIR.mkdir(parents=True, exist_ok=True)

    if args.setup:
        rc.setup_venv(HEADLINE_VENV, HEADLINE_PIN)
        rc.preflight_venv(HEADLINE_VENV, HEADLINE_PIN)
        rc.data_determinism_check()
        return 0

    # Preflight (cheap, always).
    preflight_info = rc.preflight_venv(HEADLINE_VENV, HEADLINE_PIN)
    rc.data_determinism_check()
    rc.check_load(args.force, enforce=not args.smoke)

    estimators = args.only or list(BENCH_SPECS)
    for est in estimators:
        if est not in BENCH_SPECS:
            ap.error(f"unknown estimator {est!r}")

    n_reps = 2 if args.smoke else args.reps
    results_path = SMOKE_RESULTS_PATH if args.smoke else RESULTS_PATH

    # Merge-on-write so --only reruns update a single artifact.
    payload: Dict[str, Any] = {"cells": {}}
    if results_path.exists():
        with open(results_path) as f:
            payload = json.load(f)
    payload["run_metadata"] = rc.collect_run_metadata()
    payload["headline_pin"] = HEADLINE_PIN
    payload["smoke"] = args.smoke
    fingerprint = rc.env_fingerprint(
        payload["run_metadata"], HEADLINE_PIN, python_env=preflight_info
    )
    payload["env_fingerprint"] = fingerprint

    t0 = time.time()
    for est in estimators:
        scales = (
            ["small"]
            if args.smoke and "small" in BENCH_SPECS[est]["scales"]
            else BENCH_SPECS[est]["scales"]
        )
        for scale in scales:
            cell = run_cell(est, scale, n_reps, args.smoke)
            cell["env_fingerprint"] = fingerprint
            payload["cells"][f"{est}/{scale}"] = cell
            hard = rc.hard_flags(cell["flags"])
            if hard:
                print(f"  HARD GATE FAILURE {est}/{scale}: {hard}")
            with open(results_path, "w") as f:
                json.dump(payload, f, indent=2)
            rc.cooldown(1 if args.smoke else rc.COOLDOWN_SECONDS)

    print(f"\nDone in {time.time() - t0:.0f}s -> {results_path}")
    stale = sorted(
        key for key, c in payload["cells"].items() if c.get("env_fingerprint") != fingerprint
    )
    if stale:
        print(
            "\nARTIFACT NOT PUBLICATION-READY: cells captured under a "
            f"different environment/protocol fingerprint: {stale}. "
            "Re-run them under the current environment."
        )
    failures = collect_hard_failures(payload)
    if failures:
        print(
            "\nHARD-GATE FAILURES in the artifact (including any stale "
            "merge-on-write cells; publication blocked until resolved):"
        )
        for f_ in failures:
            print(f"  - {f_}")
        return 1
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
