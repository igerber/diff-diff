# Speed Review - Practitioner Workflow Benchmarks

Scenario-driven performance measurement for end-to-end practitioner chains,
as distinct from `benchmarks/run_benchmarks.py` which measures R-parity on
isolated `fit()` calls.

## Why these exist

See [`docs/performance-scenarios.md`](../../docs/performance-scenarios.md) for
the full methodology. Short version: the existing benchmarks measure
`fit()` in isolation on 200 x 8 synthetic panels, which does not reflect what
a practitioner running the 8-step Baker et al. (2025) workflow on a real
BRFSS or geo-experiment panel actually sees. These scripts measure the full
chain (Bacon -> fit -> HonestDiD -> cross-estimator robustness -> reporting)
at data shapes anchored to applied-econ conventions.

## Layout

```
benchmarks/speed_review/
├── README.md                           # this file
├── bench_shared.py                     # timing + pyinstrument + RSS harness
├── run_all.py                          # orchestrator (both backends)
├── bench_campaign_staggered.py         # Scenario 1: CS + 8-step chain
├── bench_brand_awareness_survey.py     # Scenario 2: DiD + SurveyDesign
├── bench_brfss_panel.py                # Scenario 3: aggregate_survey -> CS
├── bench_geo_few_markets.py            # Scenario 4: SDiD + jackknife
├── bench_reversible_dcdh.py            # Scenario 5: dCDH L_max + TSL
├── bench_dose_response.py              # Scenario 6: ContinuousDiD splines
├── mem_profile_brfss.py                # tracemalloc allocator attribution
│                                       #   for BRFSS-1M (standalone)
├── bench_callaway.py                   # pre-existing CS scaling sweep
├── baseline_results.json               # pre-existing CS baseline
├── bench_memory_scaling.py             # peak-RSS sweep for the memory-scaling
│                                       #   work (B1 #561 / B2 #563 / C #567);
│                                       #   subprocess-isolated ru_maxrss, median
├── bench_fe_absorption.py              # Scenarios 7-13: MAP-demeaning hot path
│                                       #   (subprocess-isolated, multi-run CV,
│                                       #   ATT/SE identity capture + gate)
├── bench_fe_absorption_pyfixest.py     # optional external yardstick (guarded
│                                       #   on `import pyfixest`; never a dep)
├── fe_absorption_datagen.py            # seeded DGPs shared by both FE lanes
└── baselines/                          # this effort's output
    ├── memory_scaling_{before,after}.json  # peak RSS pre-#561 vs current
    ├── fe_absorption_{before,after}.json   # FE-absorption timings + identity
    ├── fe_absorption_pyfixest.json     # yardstick timings + parity (optional)
    ├── <scenario>_<backend>.json       # phase-level wall-clock + peak RSS
    ├── mem_profile_brfss_large_<backend>.txt   # tracemalloc top-N sites
    └── profiles/                       # flame HTMLs (gitignored)
        └── <scenario>_<backend>.html   # pyinstrument flame output
```

Each JSON baseline records both timing (per-phase wall-clock) and memory
(start/peak/growth from a psutil background sampler at 10 ms). The
`mem_profile_brfss.py` script does a separate tracemalloc pass on the
BRFSS-1M scenario - this is kept out of the main timing harness because
tracemalloc has 2-5x overhead and would contaminate wall-clock baselines.

**Note on profile HTMLs.** pyinstrument flames are ~500KB-1.2MB each and are
regenerated on every run; they live under `baselines/profiles/` which is
gitignored. The key hotspots identified from them are already captured in
the findings doc (top-5 hot phases per scenario); run a scenario locally
to regenerate the full flame when needed.

## Running

```bash
# One-time install
pip install pyinstrument

# All scenarios, both backends, all scales
python benchmarks/speed_review/run_all.py

# One scenario, one backend (the script runs its full scale sweep internally)
DIFF_DIFF_BACKEND=rust python benchmarks/speed_review/bench_campaign_staggered.py

# Subset
python benchmarks/speed_review/run_all.py --scenarios brfss_panel geo_few_markets
```

Multi-scale scenarios write per-scale outputs
(e.g. `campaign_staggered_small_rust.json`, `..._medium_rust.json`,
`..._large_rust.json`). Single-scale scenarios write the scale-free form
(e.g. `dose_response_rust.json`). Full runtime for all scales × both
backends is ~90 seconds on Apple Silicon M4.

### FE-absorption suite (scenarios 7-13)

Standalone like `bench_memory_scaling.py` (not part of `run_all.py` - the 5M-row
scenarios are too heavy for the routine sweep):

```bash
# Full suite (~20-40 min on M4; strictly sequential subprocesses by design)
python benchmarks/speed_review/bench_fe_absorption.py \
    --out benchmarks/speed_review/baselines/fe_absorption_before.json

# Smoke test / one scenario
python benchmarks/speed_review/bench_fe_absorption.py --quick
python benchmarks/speed_review/bench_fe_absorption.py --only geo_experiment

# After an optimization: regenerate + prove estimates did not move
python benchmarks/speed_review/bench_fe_absorption.py \
    --out benchmarks/speed_review/baselines/fe_absorption_after.json \
    --check-estimates benchmarks/speed_review/baselines/fe_absorption_before.json

# Optional external yardstick (skipped cleanly if pyfixest is absent)
pip install pyfixest
python benchmarks/speed_review/bench_fe_absorption_pyfixest.py
```

Do not run anything else on the machine during a baseline run - the committed
JSONs carry a CV field per scenario and the driver flags CV > 10% as unusable.

## Where to look for findings

[`docs/performance-plan.md`](../../docs/performance-plan.md) - "Practitioner
Workflow Baseline (v3.1.3)" section holds per-scenario hot-phase rankings
and action recommendations. The scenarios here are the measurement surface;
the findings doc is the decision output.

## Adding a scenario

1. Add the scenario definition to `docs/performance-scenarios.md`
   (persona, data shape, operation chain, source anchor).
2. Add `bench_<name>.py` following the existing scripts: build data, define
   `phases` as a list of `(label, callable)` tuples, call `run_scenario`.
3. Register it in `run_all.py`'s `SCRIPTS` dict.
4. Run under both backends and commit the refreshed `baselines/*.json`.
   The `baselines/profiles/*.html` flame HTMLs are gitignored and
   regenerated per run - do not commit them.
5. Add a per-scenario finding paragraph to `docs/performance-plan.md`.
