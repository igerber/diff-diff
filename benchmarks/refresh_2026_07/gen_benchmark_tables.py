#!/usr/bin/env python3
"""
Regenerate the numeric tables in docs/benchmarks.rst from refresh results.

Modeled on benchmarks/speed_review/gen_findings_tables.py: every regenerable
numeric surface on the published page lives between marker comments

    .. refresh-table-start: <key>
    .. refresh-table-end: <key>

and this script rewrites ONLY the content between markers, from the
committed benchmarks/refresh_2026_07/results/refresh_results.json. Prose
stays hand-written. Regions with no data in the results JSON are left
untouched (so a partial run never wipes published tables).

Usage:
    python gen_benchmark_tables.py            # rewrite docs/benchmarks.rst
    python gen_benchmark_tables.py --check    # print diff, write nothing
    python gen_benchmark_tables.py --results path.json
"""

import argparse
import difflib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REFRESH_DIR = Path(__file__).parent
REPO_ROOT = REFRESH_DIR.parent.parent
DEFAULT_RESULTS = REFRESH_DIR / "results" / "refresh_results.json"
DOCS_RST = REPO_ROOT / "docs" / "benchmarks.rst"

MARKER_START = ".. refresh-table-start: {key}"
MARKER_END = ".. refresh-table-end: {key}"

# Hard-gate classification is shared with run_refresh.py via refresh_common
# (single source of truth - the two consumers can never drift).
from refresh_common import hard_flags as _hard_flags  # noqa: E402


def assert_uniform_environment(payload: Dict[str, Any]) -> None:
    """
    Every rendered cell must come from the same environment/protocol as the
    payload's run metadata - a partial --only rerun after an R package
    upgrade (or protocol change) can never silently mix with older cells.
    """
    expected = payload.get("env_fingerprint")
    stale = sorted(
        key
        for key, cell in payload.get("cells", {}).items()
        if cell.get("env_fingerprint") != expected
    )
    if expected is None or stale:
        raise SystemExit(
            "REFUSING generation - cells captured under a different "
            f"environment/protocol than the current run metadata: "
            f"{stale or 'payload has no env_fingerprint'}. Re-run those "
            "cells (or the full refresh) so all rendered numbers share one "
            "environment."
        )


def assert_no_hard_flags(payload: Dict[str, Any]) -> None:
    """A payload with correctness-gated cells must never render as tables."""
    flagged = {}
    for key, cell in payload.get("cells", {}).items():
        hard = _hard_flags(cell.get("flags", []))
        if hard:
            flagged[key] = hard
    if flagged:
        detail = "; ".join(f"{k}: {v}" for k, v in sorted(flagged.items()))
        raise SystemExit(
            "REFUSING to generate tables - hard-gated cells present "
            f"(resolve or re-run them first): {detail}"
        )


# Ordered estimator specs: (cell key prefix, display name, R package label)
ESTIMATORS = [
    ("basic", "BasicDiD", "R fixest"),
    ("twfe", "TWFE (absorbed FE)", "R fixest"),
    ("multiperiod", "MultiPeriodDiD", "R fixest"),
    ("callaway", "CallawaySantAnna", "R did"),
    ("synthdid", "SyntheticDiD", "R synthdid"),
]

PERF_SCALES = {
    "basic": ["small", "1k", "5k", "10k", "20k"],
    "twfe": ["small", "1k", "5k", "10k", "20k"],
    "callaway": ["small", "1k", "5k", "10k", "20k"],
    "synthdid": ["small", "1k", "5k"],
}


def fmt_time(t: float) -> str:
    if t >= 100:
        return f"{t:.0f}"
    if t >= 10:
        return f"{t:.1f}"
    if t >= 1:
        return f"{t:.2f}"
    if t >= 0.01:
        return f"{t:.3f}"
    return f"{t:.4f}"


def fmt_ratio(r: float) -> str:
    if r >= 10:
        return f"**{r:.0f}x**"
    if r >= 2:
        return f"**{r:.1f}x**"
    if r >= 1:
        return f"{r:.1f}x"
    return f"{r:.1f}x"


class _MissingData(Exception):
    """Raised when a cell/arm/field needed for a region is absent."""


def _arm_median(cell: Dict[str, Any], arm: str) -> float:
    entry = cell.get(arm)
    if not entry:
        raise _MissingData(arm)
    return float(entry["timing"]["stats"]["median"])


def _arm_val(cell: Dict[str, Any], arm: str, *keys: str) -> float:
    entry = cell.get(arm)
    if not entry:
        raise _MissingData(arm)
    res = entry["result"]
    for k in keys:
        if k in res and res[k] is not None:
            return float(res[k])
    raise _MissingData(f"{arm}:{keys}")


def _list_table(widths: List[int], header: List[str], rows: List[List[str]]) -> List[str]:
    lines = [
        ".. list-table::",
        "   :header-rows: 1",
        f"   :widths: {' '.join(str(w) for w in widths)}",
        "",
    ]
    for row in [header] + rows:
        lines.append(f"   * - {row[0]}")
        for col in row[1:]:
            lines.append(f"     - {col}")
    return lines


def _cell_max_arm_diffs(cell: Dict[str, Any]) -> tuple:
    """Max ATT abs diff / SE rel diff across BOTH rendered Python arms vs R
    (the rust-only comparison would understate a larger-but-passing pure
    difference). Falls back to the comparison record if arm data is absent.
    """
    try:
        r_att = _arm_val(cell, "r", "overall_att", "att")
        r_se = _arm_val(cell, "r", "overall_se", "se")
        atts = []
        ses = []
        for arm in ("python_pure", "python_rust"):
            atts.append(abs(_arm_val(cell, arm, "overall_att", "att") - r_att))
            if r_se:
                ses.append(abs(_arm_val(cell, arm, "overall_se", "se") - r_se) / abs(r_se))
        return max(atts), max(ses) if ses else 0.0
    except _MissingData:
        return cell["comparison"]["att_diff"], cell["comparison"]["se_rel_diff"]


def gen_summary(cells: Dict[str, Any]) -> Optional[List[str]]:
    rows = []
    for key, display, _ in ESTIMATORS:
        scales = [c for k, c in cells.items() if k.startswith(f"{key}/")]
        if not scales:
            return None
        diffs = [_cell_max_arm_diffs(c) for c in scales]
        att = max(d[0] for d in diffs)
        se = max(d[1] for d in diffs)
        overlap = all(c["comparison"]["ci_overlap"] for c in scales)
        passed = all(c["comparison"]["passed"] for c in scales)
        rows.append(
            [
                display,
                f"< {att:.0e}" if att > 0 else "0",
                f"{se:.1%}",
                "Yes" if overlap else "No",
                "**PASS**" if passed else "**FAIL**",
            ]
        )
    table = _list_table(
        [25, 20, 20, 15, 20], ["Estimator", "ATT Diff", "SE Rel Diff", "CI Overlap", "Status"], rows
    )
    return table + [
        "",
        "SyntheticDiD SE differences reflect Monte Carlo dispersion of the",
        "placebo variance (R's placebo permutation is unseeded, so the two",
        "implementations agree in distribution, not draw-by-draw): its SE is",
        "gated at a 35% relative bound with R's rep-to-rep SE values recorded",
        "in the committed results artifact, while the deterministic",
        "Frank-Wolfe ATT is gated at 1e-8. All other estimators use analytical",
        "SEs gated at the tolerances above. See the SyntheticDiD methodology",
        "registry note (benchmark SE gate is Monte Carlo-bounded).",
    ]


def gen_accuracy(cells: Dict[str, Any], key: str, r_label: str) -> Optional[List[str]]:
    cell = cells.get(f"{key}/small")
    if not cell:
        return None
    try:
        return _gen_accuracy_inner(cell, r_label)
    except _MissingData:
        return None


def _gen_accuracy_inner(cell: Dict[str, Any], r_label: str) -> List[str]:
    pure_att = _arm_val(cell, "python_pure", "overall_att", "att")
    rust_att = _arm_val(cell, "python_rust", "overall_att", "att")
    r_att = _arm_val(cell, "r", "overall_att", "att")
    pure_se = _arm_val(cell, "python_pure", "overall_se", "se")
    rust_se = _arm_val(cell, "python_rust", "overall_se", "se")
    r_se = _arm_val(cell, "r", "overall_se", "se")
    t_pure = _arm_median(cell, "python_pure")
    t_rust = _arm_median(cell, "python_rust")
    t_r = _arm_median(cell, "r")

    att_diff = max(abs(pure_att - r_att), abs(rust_att - r_att))
    se_rel = max(
        abs(pure_se - r_se) / r_se if r_se else 0.0,
        abs(rust_se - r_se) / r_se if r_se else 0.0,
    )
    t_best = min(t_pure, t_rust)
    best_label = "pure" if t_pure <= t_rust else "rust"
    if t_r >= t_best:
        time_diff = f"**{t_r / t_best:.1f}x faster** ({best_label})"
    else:
        time_diff = f"{t_best / t_r:.1f}x slower ({best_label})"

    rows = [
        [
            "ATT",
            f"{pure_att:.3f}",
            f"{rust_att:.3f}",
            f"{r_att:.3f}",
            f"< {att_diff:.0e}" if att_diff > 0 else "0",
        ],
        ["SE", f"{pure_se:.3f}", f"{rust_se:.3f}", f"{r_se:.3f}", f"{se_rel:.1%}"],
        ["Time (s)", fmt_time(t_pure), fmt_time(t_rust), fmt_time(t_r), time_diff],
    ]
    return _list_table(
        [16, 21, 21, 21, 21],
        ["Metric", "diff-diff (Pure)", "diff-diff (Rust)", r_label, "Difference"],
        rows,
    )


def gen_perf(cells: Dict[str, Any], key: str) -> Optional[List[str]]:
    rows = []
    for scale in PERF_SCALES[key]:
        cell = cells.get(f"{key}/{scale}")
        if not cell:
            return None
        try:
            t_r = _arm_median(cell, "r")
            t_pure = _arm_median(cell, "python_pure")
            t_rust = _arm_median(cell, "python_rust")
        except _MissingData:
            return None
        rows.append(
            [
                scale,
                fmt_time(t_r),
                fmt_time(t_pure),
                fmt_time(t_rust),
                fmt_ratio(t_r / t_pure),
                fmt_ratio(t_r / t_rust),
            ]
        )
    return _list_table(
        [12, 15, 18, 18, 12, 12],
        ["Scale", "R (s)", "Python Pure (s)", "Python Rust (s)", "Pure/R", "Rust/R"],
        rows,
    )


def gen_mpdta(cells: Dict[str, Any]) -> Optional[List[str]]:
    cell = cells.get("mpdta/real")
    if not cell:
        return None
    try:
        py_att = _arm_val(cell, "python_rust", "overall_att", "att")
        r_att = _arm_val(cell, "r", "overall_att", "att")
        py_se = _arm_val(cell, "python_rust", "overall_se", "se")
        r_se = _arm_val(cell, "r", "overall_se", "se")
        t_py = _arm_median(cell, "python_rust")
        t_r = _arm_median(cell, "r")
    except _MissingData:
        return None
    att_diff = abs(py_att - r_att)
    n_reps = cell["python_rust"]["timing"]["n_reps"]
    rows = [
        [
            "ATT",
            f"{py_att:.6f}",
            f"{r_att:.6f}",
            "**0** (exact match)" if att_diff < 5e-7 else f"{att_diff:.1e}",
        ],
        [
            "SE (analytical)",
            f"{py_se:.4f}",
            f"{r_se:.4f}",
            f"**< {max(abs(py_se - r_se) / r_se, 0.001):.1%}**",
        ],
        [
            f"Time (median of {n_reps})",
            f"{fmt_time(t_py)}s",
            f"{fmt_time(t_r)}s",
            f"**{t_r / t_py:.1f}x faster**",
        ],
    ]
    return _list_table([25, 25, 25, 25], ["Metric", "diff-diff", "R did", "Difference"], rows)


def gen_environment(payload: Dict[str, Any]) -> Optional[List[str]]:
    meta = payload.get("run_metadata")
    cells = payload.get("cells", {})
    if not meta or not cells:
        return None
    # Pull wheel provenance from any python arm.
    prov = {}
    r_meta = {}
    for cell in cells.values():
        entry = cell.get("python_rust")
        if entry and entry["result"].get("provenance"):
            prov = entry["result"]["provenance"]
        r_entry = cell.get("r")
        if r_entry and r_entry["result"].get("metadata"):
            r_meta = r_entry["result"]["metadata"]
        if prov and r_meta:
            break
    pkgs = meta.get("r_packages", {})
    pkg_str = ", ".join(f"{k} {v}" for k, v in pkgs.items() if k in ("fixest", "did", "synthdid"))
    hw = meta["hardware"]
    lines = [
        ".. rubric:: Benchmark environment (2026-07 refresh)",
        "",
        f"- **Captured**: {meta['date_utc']}",
        f"- **Hardware**: {hw['cpu']}, {hw['memory_gb']} GB RAM, " f"{hw['os']} ({hw['arch']})",
        f"- **diff-diff**: {payload.get('headline_pin')} released wheel "
        f"from PyPI (Rust backend + Apple Accelerate), Python "
        f"{prov.get('python_version', '?')}, NumPy "
        f"{prov.get('numpy_version', '?')}, pandas "
        f"{prov.get('pandas_version', '?')}",
        f"- **R**: {meta.get('r_version', '?')}; {pkg_str} " f"(installed at capture)",
        f"- **Threads**: {meta['thread_policy']}",
        f"- **Protocol**: {meta['protocol']}",
    ]
    return lines


def gen_perf_basic_twfe(cells: Dict[str, Any]) -> Optional[List[str]]:
    """
    The perf_basic docs region renders BOTH the simple interaction OLS
    (BasicDiD) and the genuine absorbed-FE TWFE pair - the legacy page
    labeled a single interaction-OLS table "BasicDiD/TWFE".
    """
    basic = gen_perf(cells, "basic")
    twfe = gen_perf(cells, "twfe")
    if basic is None or twfe is None:
        return None
    return (
        ["**BasicDiD (interaction OLS, clustered):**", ""]
        + basic
        + ["", "**TWFE (absorbed unit + post fixed effects, clustered):**", ""]
        + twfe
    )


def build_regions(payload: Dict[str, Any]) -> Dict[str, Optional[List[str]]]:
    cells = payload.get("cells", {})
    return {
        "summary": gen_summary(cells),
        "accuracy_basic": gen_accuracy(cells, "basic", "R fixest"),
        "accuracy_multiperiod": gen_accuracy(cells, "multiperiod", "R fixest"),
        "accuracy_synthdid": gen_accuracy(cells, "synthdid", "R synthdid"),
        "accuracy_callaway": gen_accuracy(cells, "callaway", "R did"),
        "environment": gen_environment(payload),
        "perf_basic": gen_perf_basic_twfe(cells),
        "perf_callaway": gen_perf(cells, "callaway"),
        "perf_synthdid": gen_perf(cells, "synthdid"),
        "mpdta": gen_mpdta(cells),
    }


def splice(src: str, key: str, lines: List[str]) -> str:
    start = MARKER_START.format(key=key)
    end = MARKER_END.format(key=key)
    pattern = re.compile(
        re.escape(start) + r"\n.*?" + re.escape(end),
        flags=re.DOTALL,
    )
    if not pattern.search(src):
        raise SystemExit(f"marker pair not found in {DOCS_RST}: {key}")
    replacement = start + "\n\n" + "\n".join(lines) + "\n\n" + end
    return pattern.sub(lambda _: replacement, src, count=1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument(
        "--check", action="store_true", help="print diff and exit 1 if the page would change"
    )
    ap.add_argument(
        "--allow-partial",
        action="store_true",
        help="tolerate regions with no data (local partial runs); default "
        "publication mode fails so fresh and stale tables can never mix",
    )
    args = ap.parse_args()

    with open(args.results) as f:
        payload = json.load(f)

    assert_no_hard_flags(payload)
    if not args.allow_partial:
        assert_uniform_environment(payload)

    original = DOCS_RST.read_text()
    updated = original
    skipped = []
    for key, lines in build_regions(payload).items():
        if lines is None:
            skipped.append(key)
            continue
        updated = splice(updated, key, lines)
    if skipped:
        if not args.allow_partial:
            raise SystemExit(
                "REFUSING partial generation - these regions have no data in "
                f"{args.results.name}: {', '.join(skipped)}. Re-run the "
                "missing cells or pass --allow-partial for local iteration."
            )
        print(f"regions left untouched (no data): {', '.join(skipped)}", file=sys.stderr)

    if args.check:
        if updated == original:
            print("docs/benchmarks.rst is up to date")
            return 0
        diff = difflib.unified_diff(
            original.splitlines(keepends=True),
            updated.splitlines(keepends=True),
            fromfile="docs/benchmarks.rst (current)",
            tofile="docs/benchmarks.rst (regenerated)",
        )
        sys.stdout.writelines(diff)
        return 1

    DOCS_RST.write_text(updated)
    print(f"updated {DOCS_RST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
