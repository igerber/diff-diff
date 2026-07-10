#!/usr/bin/env python3
"""
Internal "optimization story" benchmark: released diff-diff 3.5.3 vs 3.7.0.

Quantifies the June-July 2026 optimization arc (Rust demean_map kernel,
solve_ols marshalling slimming, CallawaySantAnna O(n_units) aggregation,
bootstrap memory tiling) by running the SAME benchmark scripts on the SAME
seed-42 data against two pinned PyPI wheels in isolated venvs, strictly
sequentially. v3.5.3 (2026-06-25) predates the entire 3.6.x wave.

Both arms run at wheel defaults (backend auto -> Rust + Accelerate; no
DIFF_DIFF_* env knobs set) - i.e. exactly what `pip install diff-diff` users
got on each date. Note the opt-in solve_ols Cholesky fast path (#670) is in
NO released wheel and plays no role here.

Methods: MultiPeriodDiD (absorbed-FE event study - exercises the demeaning
arc) and CallawaySantAnna (exercises the CS scaling arc) at the harness 20k
scale plus one larger story-scale cell each (~2M rows); BasicDiD as a cheap
"already fast, unchanged" context cell. SyntheticDiD is deliberately
excluded: its Frank-Wolfe rework shipped in v3.3.0, before both pins.

This is a repo-internal artifact (results/version_story.{json,md}); it is
NOT part of the docs site.

Usage:
    python run_version_story.py --setup       # venvs + preflight only
    python run_version_story.py --smoke       # small-scale compat check
    python run_version_story.py               # full run (idle machine!)
    python run_version_story.py --report      # regenerate md from json
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import refresh_common as rc

sys.path.insert(0, str(rc.REPO_ROOT))

from benchmarks.run_benchmarks import SCALE_CONFIGS  # noqa: E402

PINS = {"dd353": "3.5.3", "dd370": "3.7.0"}
OLD_VENV, NEW_VENV = "dd353", "dd370"

RESULTS_JSON = rc.RESULTS_DIR / "version_story.json"
RESULTS_MD = rc.RESULTS_DIR / "version_story.md"

# Story-scale configs (~2M rows). Calibrate with --story-scale story50k
# first if unsure the old arm stays under ~10 min/rep.
STORY_CONFIGS: Dict[str, Dict[str, Dict[str, int]]] = {
    "story2m": {
        "multiperiod": {"n_units": 100_000, "n_pre": 10, "n_post": 10},
        "staggered": {"n_units": 100_000, "n_periods": 20, "n_cohorts": 8},
    },
    "story50k": {
        "multiperiod": {"n_units": 50_000, "n_pre": 10, "n_post": 10},
        "staggered": {"n_units": 50_000, "n_periods": 20, "n_cohorts": 8},
    },
}

CELLS: List[Dict[str, Any]] = [
    {
        "estimator": "basic",
        "display": "BasicDiD (interaction OLS)",
        "py_script": "benchmark_basic.py",
        "py_estimator": "diff_diff.DifferenceInDifferences",
        "dataset": "basic",
        "scale": "20k",
        "extra": ["--type", "basic"],
        "reps": 6,
        "timeout": 2400,
    },
    {
        "estimator": "multiperiod",
        "display": "MultiPeriodDiD (absorbed FE)",
        "py_script": "benchmark_multiperiod.py",
        "py_estimator": "diff_diff.MultiPeriodDiD",
        "dataset": "multiperiod",
        "scale": "20k",
        "reps": 6,
        "timeout": 2400,
    },
    {
        "estimator": "callaway",
        "display": "CallawaySantAnna",
        "py_script": "benchmark_callaway.py",
        "py_estimator": "diff_diff.CallawaySantAnna",
        "dataset": "staggered",
        "scale": "20k",
        "reps": 6,
        "timeout": 2400,
    },
    {
        "estimator": "multiperiod",
        "display": "MultiPeriodDiD (absorbed FE)",
        "py_script": "benchmark_multiperiod.py",
        "py_estimator": "diff_diff.MultiPeriodDiD",
        "dataset": "multiperiod",
        "scale": "STORY",
        "reps": 4,
        "timeout": 3600,
    },
    {
        "estimator": "callaway",
        "display": "CallawaySantAnna",
        "py_script": "benchmark_callaway.py",
        "py_estimator": "diff_diff.CallawaySantAnna",
        "dataset": "staggered",
        "scale": "STORY",
        "reps": 4,
        "timeout": 3600,
    },
]


def dataset_for(cell: Dict[str, Any], story_scale: str) -> Path:
    scale = cell["scale"]
    if scale == "STORY":
        cfg = STORY_CONFIGS[story_scale][cell["dataset"]]
        return rc.ensure_dataset(cell["dataset"], story_scale, cfg)
    cfg = SCALE_CONFIGS[scale][cell["dataset"]]
    return rc.ensure_dataset(cell["dataset"], scale, cfg)


def cell_extra(cell: Dict[str, Any], story_scale: str) -> List[str]:
    extra = list(cell.get("extra", []))
    if cell["estimator"] == "multiperiod":
        scale = cell["scale"]
        cfg = (
            STORY_CONFIGS[story_scale]["multiperiod"]
            if scale == "STORY"
            else SCALE_CONFIGS[scale]["multiperiod"]
        )
        extra += ["--n-pre", str(cfg["n_pre"]), "--n-post", str(cfg["n_post"])]
    return extra


def run_story_cell(
    cell: Dict[str, Any],
    story_scale: str,
    n_reps: int,
    smoke: bool,
) -> Dict[str, Any]:
    data_path = dataset_for(cell, story_scale)
    extra = cell_extra(cell, story_scale)
    scale_label = story_scale if cell["scale"] == "STORY" else cell["scale"]
    out: Dict[str, Any] = {
        "estimator": cell["display"],
        "scale": scale_label,
        "data_file": data_path.name,
        "data_sha256": rc.sha256_file(data_path),
        "flags": [],
    }
    print(
        f"\n=== STORY {cell['display']} @ {scale_label} "
        f"(reps={n_reps}, data={data_path.name}) ==="
    )

    for venv in (OLD_VENV, NEW_VENV):
        pin = PINS[venv]
        raw_out = rc.RAW_DIR / (f"story_{cell['estimator']}_{scale_label}_{venv}.json")

        def rep_fn(venv=venv, pin=pin, raw_out=raw_out):
            res = rc.run_python_rep(
                cell["py_script"],
                data_path,
                raw_out,
                backend="auto",
                venv=venv,
                extra_args=extra,
                timeout=cell["timeout"],
            )
            # Wheel default = auto = rust backend on both pins.
            rc.validate_python_provenance(res, pin, venv, "rust")
            return res

        try:
            arm = rc.run_arm(
                f"{cell['estimator']}/{scale_label}/{pin}",
                rep_fn,
                n_reps,
                allow_cv_rerun=not smoke,
            )
        except Exception as exc:  # old-wheel API drift: drop, never hack
            print(f"    [{pin}] CELL DROPPED: {exc}")
            out["flags"].append(f"{pin}:dropped:{type(exc).__name__}")
            out[pin] = {"dropped": True, "reason": str(exc)[:500]}
            continue
        rc.validate_estimator_field(arm["result"], cell["py_estimator"])
        arm["result"] = rc.slim_result(arm["result"])
        out[pin] = arm
        out["flags"].extend(f"{pin}:{fl}" for fl in arm["flags"])
        rc.cooldown(1 if smoke else rc.COOLDOWN_SECONDS)

    # Cross-version invariants (balanced fixtures: estimates must agree).
    old, new = out.get(PINS[OLD_VENV]), out.get(PINS[NEW_VENV])
    if old and new and not old.get("dropped") and not new.get("dropped"):

        def _att(entry):
            res = entry["result"]
            return float(res.get("overall_att", res.get("att")))

        gap = abs(_att(old) - _att(new))
        out["att_gap_old_new"] = gap
        if gap > 1e-6:
            out["flags"].append(f"cross_version_att_shift:{gap:.2e}")
        for lib in ("numpy_version", "pandas_version"):
            vo = old["result"].get("provenance", {}).get(lib)
            vn = new["result"].get("provenance", {}).get(lib)
            if vo != vn:
                out["flags"].append(f"venv_dep_mismatch:{lib}:{vo}!={vn}")
    return out


def write_report(payload: Dict[str, Any]) -> None:
    meta = payload["run_metadata"]
    flagged = {k: c["flags"] for k, c in payload["cells"].items() if c.get("flags")}
    lines = [
        "# diff-diff optimization story: released 3.5.3 vs 3.7.0",
        "",
    ]
    if flagged:
        lines += [
            "> **ATTENTION - flagged cells below; do not quote these numbers",
            "> without resolving the flags:**",
            ">",
        ]
        for k, fls in flagged.items():
            lines.append(f"> - `{k}`: {', '.join(fls)}")
        lines.append("")
    lines += [
        "Internal benchmark artifact (NOT part of the docs site). Same",
        "benchmark scripts, same seed-42 data, two pinned PyPI wheels in",
        "isolated venvs, wheel defaults (Rust + Accelerate backend, no",
        "DIFF_DIFF_* env knobs), fresh subprocess per replication, strictly",
        "sequential, untimed in-process warm-up, median of counted reps",
        "(first rep excluded).",
        "",
        "v3.5.3 (2026-06-25) predates the June-July optimization arc;",
        "v3.7.0 (2026-07-08) contains it (Rust demean_map FE-absorption",
        "kernel, solve_ols marshalling slimming, CallawaySantAnna O(n_units)",
        "aggregation + fused bootstrap, multiplier-bootstrap memory tiling).",
        "The opt-in solve_ols Cholesky fast path (#670) is in NO released",
        "wheel and plays no role in these numbers. SyntheticDiD is excluded:",
        "its Frank-Wolfe rework shipped in v3.3.0, before both pins.",
        "",
        "| Estimator | Scale | Rows | 3.5.3 median (s) | 3.7.0 median (s) " "| Speedup |",
        "|---|---|---|---|---|---|",
    ]
    for key, cell in payload["cells"].items():
        old = cell.get("3.5.3", {})
        new = cell.get("3.7.0", {})
        if old.get("dropped") or new.get("dropped") or not old or not new:
            lines.append(
                f"| {cell['estimator']} | {cell['scale']} | - | - | - | "
                f"DROPPED ({'; '.join(cell['flags'])}) |"
            )
            continue
        t_old = old["timing"]["stats"]["median"]
        t_new = new["timing"]["stats"]["median"]
        rows = old["result"].get("metadata", {}).get("n_obs", "-")
        speed = t_old / t_new if t_new else float("nan")
        lines.append(
            f"| {cell['estimator']} | {cell['scale']} | {rows:,} "
            f"| {t_old:.3f} | {t_new:.3f} | **{speed:.2f}x** |"
            if isinstance(rows, int)
            else f"| {cell['estimator']} | {cell['scale']} | {rows} "
            f"| {t_old:.3f} | {t_new:.3f} | **{speed:.2f}x** |"
        )
    lines += [
        "",
        "## Environment",
        "",
        f"- Captured: {meta['date_utc']}",
        f"- Hardware: {meta['hardware']['cpu']}, "
        f"{meta['hardware']['memory_gb']} GB, {meta['hardware']['os']} "
        f"({meta['hardware']['arch']})",
        f"- Wheels: diff-diff 3.5.3 and 3.7.0 (PyPI macosx-arm64, Rust "
        f"backend + Apple Accelerate), python {meta['orchestrator_python']}",
        f"- Protocol: {meta['protocol']}",
        f"- Thread policy: {meta['thread_policy']}",
        "",
        "Flags (if any) per cell:",
        "",
    ]
    for key, cell in payload["cells"].items():
        if cell.get("flags"):
            lines.append(f"- `{key}`: {', '.join(cell['flags'])}")
    if not any(c.get("flags") for c in payload["cells"].values()):
        lines.append("- none")
    lines.append("")
    RESULTS_MD.write_text("\n".join(lines))
    print(f"wrote {RESULTS_MD}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--setup", action="store_true")
    ap.add_argument(
        "--smoke", action="store_true", help="small-scale old-wheel compat check (reps=2)"
    )
    ap.add_argument(
        "--report", action="store_true", help="regenerate version_story.md from existing json"
    )
    ap.add_argument(
        "--story-scale",
        default="story2m",
        choices=list(STORY_CONFIGS) + ["none"],
        help="size of the large story cells (default story2m; "
        "use story50k to calibrate, none to skip them)",
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    rc.RAW_DIR.mkdir(parents=True, exist_ok=True)

    if args.report:
        with open(RESULTS_JSON) as f:
            write_report(json.load(f))
        return 0

    if args.setup:
        for venv, pin in PINS.items():
            rc.setup_venv(venv, pin)
            rc.preflight_venv(venv, pin)
        return 0

    for venv, pin in PINS.items():
        rc.preflight_venv(venv, pin)
    rc.check_load(args.force, enforce=not args.smoke)

    if args.smoke:
        # Compat guard: every story script x the OLD wheel at small scale.
        cells = [
            {**c, "scale": "small", "reps": 2, "timeout": 600}
            for c in CELLS
            if c["scale"] == "20k"  # one per script, small data
        ]
    else:
        cells = [c for c in CELLS if not (c["scale"] == "STORY" and args.story_scale == "none")]

    payload: Dict[str, Any] = {"cells": {}}
    if RESULTS_JSON.exists() and not args.smoke:
        with open(RESULTS_JSON) as f:
            payload = json.load(f)
    payload["run_metadata"] = rc.collect_run_metadata()
    payload["pins"] = PINS

    t0 = time.time()
    for cell in cells:
        scale_label = args.story_scale if cell["scale"] == "STORY" else cell["scale"]
        n_reps = 2 if args.smoke else cell["reps"]
        out = run_story_cell(cell, args.story_scale, n_reps, args.smoke)
        payload["cells"][f"{cell['estimator']}/{scale_label}"] = out
        if not args.smoke:
            with open(RESULTS_JSON, "w") as f:
                json.dump(payload, f, indent=2)
        rc.cooldown(1 if args.smoke else rc.COOLDOWN_SECONDS)

    print(f"\nStory run done in {time.time() - t0:.0f}s")
    flagged = {k: c["flags"] for k, c in payload["cells"].items() if c.get("flags")}
    if not args.smoke:
        with open(RESULTS_JSON, "w") as f:
            json.dump(payload, f, indent=2)
        write_report(payload)
    else:
        print("(smoke mode: no artifacts written)")
    if flagged:
        print("\nFLAGGED CELLS (report banner added; exit nonzero):")
        for k, fls in flagged.items():
            print(f"  - {k}: {fls}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
