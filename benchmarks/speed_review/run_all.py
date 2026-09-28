#!/usr/bin/env python3
"""
Run every practitioner-workflow scenario under both backends.

Writes per-scenario JSON + pyinstrument HTML under
``benchmarks/speed_review/baselines/`` (and ``.../baselines/profiles/``).
See ``docs/performance-scenarios.md`` for scenario definitions and
``docs/performance-plan.md`` for the derived findings.

Exit status is nonzero if any scenario subprocess exits nonzero. Scenario
scripts themselves exit 1 on any phase failure (see ``bench_shared.py``),
so this orchestrator reliably surfaces failures.

Usage:

    python benchmarks/speed_review/run_all.py
    python benchmarks/speed_review/run_all.py --backend python
    python benchmarks/speed_review/run_all.py --backend rust
    python benchmarks/speed_review/run_all.py --scenarios campaign_staggered
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = {
    "campaign_staggered": "bench_campaign_staggered.py",
    "brand_awareness_survey": "bench_brand_awareness_survey.py",
    "brfss_panel": "bench_brfss_panel.py",
    "geo_few_markets": "bench_geo_few_markets.py",
    "reversible_dcdh": "bench_reversible_dcdh.py",
    "dose_response": "bench_dose_response.py",
}


def run(scenario, backend):
    script = HERE / SCRIPTS[scenario]
    env = os.environ.copy()
    env["DIFF_DIFF_BACKEND"] = backend
    print(f"\n===== {scenario} backend={backend} =====")
    result = subprocess.run(
        [sys.executable, str(script)], env=env,
    )
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", choices=["python", "rust", "both"], default="both",
    )
    parser.add_argument(
        "--scenarios", nargs="+", choices=list(SCRIPTS),
        default=list(SCRIPTS),
    )
    args = parser.parse_args()

    if args.backend == "both":
        backends = ["python", "rust"]
    else:
        backends = [args.backend]

    failures = []
    for backend in backends:
        for scenario in args.scenarios:
            if not run(scenario, backend):
                failures.append((scenario, backend))

    print("\n\n===== SUMMARY =====")
    if failures:
        print(f"{len(failures)} scenario/backend combos failed:")
        for s, b in failures:
            print(f"  - {s} ({b})")
        sys.exit(1)
    else:
        print("All scenarios passed.")


if __name__ == "__main__":
    main()
