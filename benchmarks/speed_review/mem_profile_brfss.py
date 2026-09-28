"""
Per-function allocation attribution for the BRFSS-1M scenario.

Runs the large-scale BRFSS `aggregate_survey` path under `tracemalloc` and
writes top-N allocation sites to
``benchmarks/speed_review/baselines/mem_profile_brfss_large_<backend>.txt``.

Standalone because tracemalloc has 2-5x overhead; running it inside the
main timing harness would contaminate the wall-clock baselines. Companion
to the `resource.getrusage`-style peak RSS captured in the main JSON
baselines — this script tells us WHERE the memory went, those tell us
HOW MUCH.
"""

import argparse
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

from diff_diff import SurveyDesign, aggregate_survey
from diff_diff._backend import HAS_RUST_BACKEND


BASELINES = Path(__file__).resolve().parent / "baselines"


def build_microdata(n_states=50, n_years=10, n_per_cell=2000,
                    n_strata=20, n_psu=1000, seed=42):
    rng = np.random.default_rng(seed)
    n_rows = n_states * n_years * n_per_cell
    state = np.repeat(np.arange(n_states), n_years * n_per_cell)
    year = np.tile(
        np.repeat(np.arange(2010, 2010 + n_years), n_per_cell),
        n_states,
    )
    stratum = rng.integers(0, n_strata, size=n_rows)
    psu = stratum * (n_psu // n_strata) + rng.integers(
        0, n_psu // n_strata, size=n_rows,
    )
    weight = rng.lognormal(0, 0.4, size=n_rows) * 50.0
    y = (
        rng.normal(0, 1, size=n_rows)
        + 0.5 * (year - 2010)
        + rng.normal(0, 0.2, size=n_rows) * state
    )
    return pd.DataFrame({
        "state": state, "year": year,
        "strata": stratum, "psu": psu, "finalwt": weight,
        "y": y,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=15,
                        help="Show top N allocation sites")
    args = parser.parse_args()

    BASELINES.mkdir(parents=True, exist_ok=True)
    backend = "rust" if HAS_RUST_BACKEND else "python"
    out_path = BASELINES / f"mem_profile_brfss_large_{backend}.txt"

    print("Building 1M-row BRFSS microdata...")
    micro = build_microdata()
    print(f"  shape: {micro.shape}, mem: "
          f"{micro.memory_usage(deep=True).sum()/1024/1024:.1f} MB")

    sd = SurveyDesign(
        weights="finalwt", strata="strata", psu="psu",
    )

    print("Starting tracemalloc...")
    tracemalloc.start(25)
    snap_before = tracemalloc.take_snapshot()

    print("Running aggregate_survey...")
    panel, stage2 = aggregate_survey(
        micro, by=["state", "year"], outcomes="y",
        survey_design=sd,
    )

    snap_after = tracemalloc.take_snapshot()
    stats = snap_after.compare_to(snap_before, "lineno")
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    total_net_diff = sum(s.size_diff for s in stats) / (1024 * 1024)
    top_site_diff = (
        stats[0].size_diff / (1024 * 1024) if stats else 0.0
    )

    lines = [
        f"# BRFSS-1M aggregate_survey allocation attribution",
        f"# backend: {backend}",
        f"# input microdata rows: {len(micro):,}",
        f"# input microdata memory: "
        f"{micro.memory_usage(deep=True).sum()/1024/1024:.1f} MB",
        f"# output panel cells: {len(panel)}",
        f"",
        f"# tracemalloc totals during aggregate_survey",
        f"# total net size diff across all sites: {total_net_diff:.1f} MB",
        f"# top single-site size diff: {top_site_diff:.2f} MB",
        f"# python peak traced: {peak/1024/1024:.1f} MB",
        f"# python current retained: {current/1024/1024:.1f} MB",
        f"",
        f"# top {args.top} allocation sites by size delta",
        f"{'#':<4} {'size diff (MB)':>16} {'count diff':>12}  location",
        f"{'-'*80}",
    ]
    # Scrub workstation-specific absolute paths before committing output
    # (keeps the file reproducible and avoids leaking $HOME / system paths).
    import site, sys as _sys
    home = str(Path.home())
    sys_paths = sorted(
        {p for p in (site.getsitepackages() + [site.getusersitepackages()])
         if p} | {_sys.prefix, _sys.base_prefix},
        key=len, reverse=True,
    )
    repo_root = str(Path(__file__).resolve().parents[2])

    def _scrub(s):
        s = s.replace(repo_root, "<repo>")
        for sp in sys_paths:
            s = s.replace(sp, "<site-packages>")
        s = s.replace(home, "$HOME")
        return s

    for i, s in enumerate(stats[:args.top], 1):
        loc = _scrub(str(s.traceback).split("\n")[0])
        lines.append(
            f"{i:<4} {s.size_diff/1024/1024:>16.2f} {s.count_diff:>12d}  {loc}"
        )

    text = "\n".join(lines) + "\n"
    out_path.write_text(text)
    print("\n" + text)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
