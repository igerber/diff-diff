# diff-diff optimization story: released 3.5.3 vs 3.7.0

**Headline: CallawaySantAnna runs ~4.9x faster on the
3.7.0 wheel than on 3.5.3** at every measured scale
(360,000 through 2,000,000 rows), with identical estimates. This is the
flagship staggered-adoption estimator and the primary target
of the June-July 2026 scaling arc (O(n_units) aggregation
influence functions, fused bootstrap, per-cell solver work).
The BasicDiD/MultiPeriodDiD rows below are supporting
context: those paths were already fast and saw incremental
gains at these benchmark shapes.

Internal benchmark artifact (NOT part of the docs site). Same
benchmark scripts, same seed-42 data, two pinned PyPI wheels in
isolated venvs, wheel defaults (Rust + Accelerate backend, no
DIFF_DIFF_* env knobs), fresh subprocess per replication, strictly
sequential, untimed in-process warm-up, median of counted reps
(first rep excluded).

v3.5.3 (2026-06-25) predates the June-July optimization arc;
v3.7.0 (2026-07-08) contains it (Rust demean_map FE-absorption
kernel, solve_ols marshalling slimming, CallawaySantAnna O(n_units)
aggregation + fused bootstrap, multiplier-bootstrap memory tiling).
The opt-in solve_ols Cholesky fast path (#670) is in NO released
wheel and plays no role in these numbers. SyntheticDiD is excluded:
its Frank-Wolfe rework shipped in v3.3.0, before both pins.

| Estimator | Scale | Rows | 3.5.3 median (s) | 3.7.0 median (s) | Speedup |
|---|---|---|---|---|---|
| CallawaySantAnna | 20k | 360,000 | 0.242 | 0.048 | **5.02x** |
| CallawaySantAnna | story2m | 2,000,000 | 1.335 | 0.274 | **4.87x** |
| BasicDiD (interaction OLS) | 20k | 240,000 | 0.029 | 0.022 | **1.31x** |
| MultiPeriodDiD (absorbed FE) | 20k | 320,000 | 0.439 | 0.381 | **1.15x** |
| MultiPeriodDiD (absorbed FE) | story2m | 2,000,000 | 3.542 | 2.995 | **1.18x** |

## Environment

- Captured: 2026-07-11T00:59:47+00:00
- Hardware: Apple M4 Max, 36 GB, macOS 26.5.2 (arm64)
- Wheels: diff-diff 3.5.3 and 3.7.0 (PyPI macosx-arm64, Rust backend + Apple Accelerate), python 3.14.4
- Protocol: Each replication is a fresh subprocess run strictly sequentially (one benchmark process on the machine at a time) with an untimed in-process warm-up fit before the timed fit. The first replication is additionally excluded from statistics. Published statistic: median of the counted replications. Arms with CV > 10% are rerun once and flagged if still noisy.
- Thread policy: No arm is thread-restricted: R runs at fixest/data.table defaults; diff-diff wheels run at Accelerate/rayon defaults. Thread-count env vars (RAYON/OMP/OPENBLAS/VECLIB/MKL/data.table) are stripped from every benchmark subprocess and R runs under --vanilla (no user/site .Rprofile/.Renviron), so package defaults are enforced, not assumed. Per-arm thread counts are recorded in each result's metadata.

Flags (if any) per cell:

- none
