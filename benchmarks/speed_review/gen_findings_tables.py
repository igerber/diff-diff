#!/usr/bin/env python3
"""
Regenerate the numerical tables in ``docs/performance-plan.md`` from the
committed JSON baselines under ``benchmarks/speed_review/baselines/``.

Each auto-generated table is bounded by a pair of HTML-comment markers in
the target markdown file:

    <!-- TABLE:start <table-id> -->
    ... (rendered table body lives here; overwritten on every run) ...
    <!-- TABLE:end <table-id> -->

Run this after any benchmark rerun; the doc tables then re-derive exactly
from the JSON baselines, removing the possibility of hand-edit drift.

Tables owned by this generator:
  - scale_sweep_totals        end-to-end wall-clock per scenario + scale
  - memory_by_scenario        peak RSS + growth per scenario + scale
  - top_phases_by_scenario    largest-scale phase-level timing ranking

Narrative prose in the doc is hand-written and not touched. If numerical
claims in narrative drift from the regenerated tables, the reviewer must
update the narrative manually - by design, to force a human read of the
findings whenever numbers shift meaningfully.
"""

import json
import re
from pathlib import Path
from textwrap import dedent

HERE = Path(__file__).resolve().parent
BASELINES = HERE / "baselines"
PLAN_MD = HERE.parent.parent / "docs" / "performance-plan.md"

SCALE_ORDER = ("small", "medium", "large")
MULTI_SCALE = (
    "campaign_staggered",
    "brand_awareness_survey",
    "brfss_panel",
    "geo_few_markets",
)
SINGLE_SCALE = ("reversible_dcdh", "dose_response")

SCENARIO_DISPLAY = {
    "campaign_staggered": "1. Staggered campaign",
    "brand_awareness_survey": "2. Brand awareness survey",
    "brfss_panel": "3. BRFSS microdata -> CS panel",
    "geo_few_markets": "4. SDiD few markets",
    "reversible_dcdh": "5. Reversible dCDH",
    "dose_response": "6. Pricing dose-response",
}


def load(scenario, scale, backend):
    if scale is None:
        path = BASELINES / f"{scenario}_{backend}.json"
    else:
        path = BASELINES / f"{scenario}_{scale}_{backend}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def fmt_secs(x):
    return f"{x:.2f}" if x is not None else "skip"


def fmt_mb(x):
    return f"{x:.0f}" if x is not None else "skip"


def render_scale_sweep_totals():
    rows = [
        "| Scenario | Scale | Python (s) | Rust (s) | Py/Rust |",
        "|---|---|---:|---:|---:|",
    ]
    for scen in MULTI_SCALE:
        display = SCENARIO_DISPLAY[scen]
        first = True
        for scale in SCALE_ORDER:
            py = load(scen, scale, "python")
            rs = load(scen, scale, "rust")
            py_t = py["total_seconds"] if py else None
            rs_t = rs["total_seconds"] if rs else None
            ratio = (
                f"{py_t/rs_t:.1f}x" if (py_t is not None and rs_t is not None and rs_t > 0) else "-"
            )
            name_col = display if first else ""
            first = False
            rows.append(
                f"| {name_col} | {scale} | " f"{fmt_secs(py_t)} | {fmt_secs(rs_t)} | {ratio} |"
            )
    for scen in SINGLE_SCALE:
        display = SCENARIO_DISPLAY[scen]
        py = load(scen, None, "python")
        rs = load(scen, None, "rust")
        py_t = py["total_seconds"] if py else None
        rs_t = rs["total_seconds"] if rs else None
        ratio = f"{py_t/rs_t:.1f}x" if (py_t is not None and rs_t is not None and rs_t > 0) else "-"
        rows.append(f"| {display} | single | " f"{fmt_secs(py_t)} | {fmt_secs(rs_t)} | {ratio} |")
    return "\n".join(rows)


def render_memory_by_scenario():
    rows = [
        "| Scenario | Scale | Py peak RSS (MB) | Py growth (MB) | "
        "Rust peak RSS (MB) | Rust growth (MB) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for scen in MULTI_SCALE:
        display = SCENARIO_DISPLAY[scen]
        first = True
        for scale in SCALE_ORDER:
            py = load(scen, scale, "python")
            rs = load(scen, scale, "rust")
            py_peak = py["memory"]["peak_mb"] if py else None
            py_growth = py["memory"]["growth_mb"] if py else None
            rs_peak = rs["memory"]["peak_mb"] if rs else None
            rs_growth = rs["memory"]["growth_mb"] if rs else None
            name_col = display if first else ""
            first = False
            rows.append(
                f"| {name_col} | {scale} | "
                f"{fmt_mb(py_peak)} | {fmt_mb(py_growth)} | "
                f"{fmt_mb(rs_peak)} | {fmt_mb(rs_growth)} |"
            )
    for scen in SINGLE_SCALE:
        display = SCENARIO_DISPLAY[scen]
        py = load(scen, None, "python")
        rs = load(scen, None, "rust")
        py_peak = py["memory"]["peak_mb"] if py else None
        py_growth = py["memory"]["growth_mb"] if py else None
        rs_peak = rs["memory"]["peak_mb"] if rs else None
        rs_growth = rs["memory"]["growth_mb"] if rs else None
        rows.append(
            f"| {display} | single | "
            f"{fmt_mb(py_peak)} | {fmt_mb(py_growth)} | "
            f"{fmt_mb(rs_peak)} | {fmt_mb(rs_growth)} |"
        )
    return "\n".join(rows)


def render_top_phases_by_scenario():
    """Top-3 phases at the largest-available scale per (scenario, backend).

    If a scenario/backend skips `large` (e.g., geo_few_markets Python),
    this falls back to the largest measured scale for that backend so
    the table still reports the Python-vs-Rust comparison rather than
    dropping the row entirely.
    """
    rows = [
        "| Scenario | Scale | Backend | Top phase (%) " "| 2nd phase (%) | 3rd phase (%) |",
        "|---|---|---|---|---|---|",
    ]

    def phase_rank(record, n=3):
        if record is None:
            return []
        total = record["total_seconds"]
        phases = sorted(
            record["phases"].items(),
            key=lambda kv: -kv[1]["seconds"],
        )
        out = []
        for label, info in phases[:n]:
            pct = 100 * info["seconds"] / total if total > 0 else 0
            out.append(f"`{label}` ({pct:.0f}%)")
        while len(out) < n:
            out.append("-")
        return out

    def largest_available(scen, backend):
        """Return (scale, record) for the largest scale this backend has."""
        for scale in reversed(SCALE_ORDER):
            rec = load(scen, scale, backend)
            if rec is not None:
                return scale, rec
        return None, None

    for scen in MULTI_SCALE:
        display = SCENARIO_DISPLAY[scen]
        for backend in ("python", "rust"):
            scale, rec = largest_available(scen, backend)
            top = phase_rank(rec)
            if not top:
                continue
            rows.append(f"| {display} | {scale} | {backend} | " f"{top[0]} | {top[1]} | {top[2]} |")
    for scen in SINGLE_SCALE:
        display = SCENARIO_DISPLAY[scen]
        for backend in ("python", "rust"):
            rec = load(scen, None, backend)
            top = phase_rank(rec)
            if not top:
                continue
            rows.append(f"| {display} | single | {backend} | " f"{top[0]} | {top[1]} | {top[2]} |")
    return "\n".join(rows)


FE_SCENARIO_DISPLAY = {
    "county_policy": "7. County policy event study (SunAbraham)",
    "firm_churn": "8. Firm panel with churn (SunAbraham)",
    "scanner_twfe": "9. Scanner store-week (TWFE)",
    "geo_experiment": "10. Geo experiment 5M orders (DiD absorb)",
    "survey_absorb": "11. Survey BRR replicates (DiD absorb)",
    "tail_stress": "12. Correlated-FE stress (DiD absorb)",
    "guard_small": "13. Small-panel guard (TWFE)",
}


def render_fe_absorption():
    """Before/after/yardstick wall-clock for the FE-absorption suite.

    Renders whichever of fe_absorption_before.json / fe_absorption_after.json
    / fe_absorption_pyfixest.json exist; absent files render as "-" columns so
    the table is valid from PR-A (before only) onward.
    """

    def load_suite(name):
        path = BASELINES / f"{name}.json"
        if not path.exists():
            return {}
        return {
            r["scenario"]: r for r in json.loads(path.read_text())["results"] if "error" not in r
        }

    before = load_suite("fe_absorption_before")
    after = load_suite("fe_absorption_after")
    yard = load_suite("fe_absorption_pyfixest")

    def cell(rec):
        if not rec or rec.get("fit_median_s") is None:
            return "-"
        quals = []
        # The 10% threshold mirrors CV_FLAG in bench_fe_absorption.py. The
        # diff-diff lane pre-computes `noisy`; the pyfixest lane records only
        # `fit_cv`, so derive the flag here for both.
        if rec.get("noisy") or (rec.get("fit_cv") or 0) > 0.10:
            quals.append("noisy")
        # Yardstick records with no coefficient are timing-only proxies
        # (saturated i(rel_time) stand-in for Sun-Abraham; see the footnote).
        if "coef" in rec and rec.get("coef") is None:
            quals.append("proxy")
        qual_txt = f", {', '.join(quals)}" if quals else ""
        return f"{rec['fit_median_s']:.3f} (cv {rec['fit_cv']:.1%}{qual_txt})"

    rows = [
        "| Scenario | n rows | Before (s) | After (s) | Speedup | pyfixest (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for scen, display in FE_SCENARIO_DISPLAY.items():
        b, a, y = before.get(scen), after.get(scen), yard.get(scen)
        n_obs = (b or a or y or {}).get("n_obs")
        n_col = f"{n_obs:,}" if n_obs else "-"
        speedup = (
            f"{b['fit_median_s'] / a['fit_median_s']:.1f}x"
            if (b and a and a.get("fit_median_s"))
            else "-"
        )
        rows.append(f"| {display} | {n_col} | {cell(b)} | {cell(a)} | {speedup} | {cell(y)} |")
    rows.append("")
    rows.append(
        "*noisy* = CV above the 10% unusable threshold from the noise protocol. "
        "*proxy* = timing-only pyfixest stand-in: the Sun-Abraham scenarios run a "
        "saturated `i(rel_time)` event study there (pyfixest 0.60 has no `sunab()`) - "
        "comparable demeaning load, different estimand, so those cells are not "
        "exact-estimand comparisons (see `bench_fe_absorption_pyfixest.py`)."
    )
    return "\n".join(rows)


TABLES = {
    "scale_sweep_totals": render_scale_sweep_totals,
    "memory_by_scenario": render_memory_by_scenario,
    "top_phases_by_scenario": render_top_phases_by_scenario,
    "fe_absorption": render_fe_absorption,
}


def update_markdown(path):
    text = path.read_text()
    for table_id, renderer in TABLES.items():
        body = renderer()
        pattern = re.compile(
            rf"(<!-- TABLE:start {re.escape(table_id)} -->)"
            rf".*?"
            rf"(<!-- TABLE:end {re.escape(table_id)} -->)",
            re.DOTALL,
        )
        replacement = f"\\g<1>\n{body}\n\\g<2>"
        new_text, n = pattern.subn(replacement, text)
        if n == 0:
            raise RuntimeError(
                f"No marker pair found for table '{table_id}' in {path}."
                f" Add <!-- TABLE:start {table_id} --> ..."
                f" <!-- TABLE:end {table_id} --> to the document first."
            )
        if n > 1:
            raise RuntimeError(f"Multiple marker pairs for '{table_id}' in {path}.")
        text = new_text
    path.write_text(text)


def main():
    update_markdown(PLAN_MD)
    print(f"regenerated tables in {PLAN_MD.relative_to(PLAN_MD.parents[2])}")
    for k in TABLES:
        print(f"  - {k}")


if __name__ == "__main__":
    main()
