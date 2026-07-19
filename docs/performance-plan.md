# Performance Improvement Plan

This document outlines the strategy for improving diff-diff's performance on large datasets, particularly for BasicDiD/TWFE and CallawaySantAnna estimators.

---

## Opt-in solve_ols normal-equations Cholesky fast path (v3.7, 2026-07)

`solve_ols` is the universal OLS entry point; both backends run an equilibrated
thin SVD (gelsd-parity) by default — deliberate defense-in-depth (the SVD's
singular-value truncation is a second, independent near-collinearity detector).
After the Phase 3 stage-0 Gram certification removed the pivoted-QR cost, the
SVD solve itself became the dominant stage on solver-bound fits (2026-07-10
attribution, M4 Max, rust+Accelerate):

| Scenario | Fit (median) | rust `solve_ols` | share |
|---|---|---|---|
| county (SunAbraham, 177k rows, ~100 cols) | 1.12 s | 0.65 s | 58% |
| firm (SunAbraham, 2.4M rows, ~130 cols) | 17.6 s | 9.07 s | 54% |
| scanner (TWFE, 3.25M rows) | 0.52 s | 0.12 s | 25% |
| cs_cov40 (CS dr, 2M rows, 40 cov, 95 cells) | 4.28 s | 1.66 s | 40% |

**The lever (opt-in only, per the 2026-07 correctness-first decision):**
`DIFF_DIFF_SOLVE_OLS_FASTPATH=1` (set to a positive integer; read per call so a
process can A/B) routes certified-well-conditioned full-rank solves through an
equilibrated normal-equations Cholesky instead of the SVD:

- **Python twin**: reuses the stage-0 rank-certification artifacts (the Gram is
  already built to certify rank — the marginal solve cost is X'y + a k×k
  `cho_factor`/`cho_solve`), gated by LAPACK `dpocon` at reciprocal condition
  1e-6 on the equilibrated Gram (the `_IRLS_CHOL_RCOND_GUARD` budget: forward
  error ~eps·cond ≈ 2e-10). Self-builds and self-certifies on the
  `skip_rank_check` route (no stage-0 artifacts exist there — factorization
  success alone is NOT a certificate).
- **Rust kernel** (`solve_ols_chol`, separate pyfunction so a stale extension
  degrades gracefully): self-certifying faer `Llt` with the exact 1-norm rcond,
  whose inverse byproduct doubles as the sandwich bread; one owned n×k buffer
  (the equilibrated copy, reused in place for the vcov scores — no thin-SVD U
  transient); faer-only heavy ops, so the accelerate/openblas/no-BLAS wheel
  variants behave identically.
- **Fallback contract**: any certification decline falls through the UNCHANGED
  legacy chain (Rust SVD, then gelsd), so knob-on-decline output equals
  knob-off exactly; the knob-off default is byte-identical to the pre-change
  tree (pinned by a pristine-base-tree identity gate in the benchmark lane and
  spy + exact-equality tests) with one deliberate exception: the saturated
  n == k silent-Inf vcov leak on the legacy Rust path was fixed alongside
  (CHANGELOG Fixed entry; the sentinel changes Inf -> NaN + warning — no
  benchmark scenario is saturated, so the identity gate is unaffected).

**Measured** (`benchmarks/speed_review/bench_solve_ols_fastpath.py`,
subprocess-per-scenario, medians; deltas are knob-on vs knob-off):

| Scenario | off | on | speedup | max deltas |
|---|---|---|---|---|
| county_policy (SunAbraham) | 1.13 s | 0.63 s | **1.80x** | dATT 5e-16, dSE/SE 9e-14 |
| firm_churn (SunAbraham) | 16.77 s | 9.89 s | **1.70x** | dATT 9e-15, dSE/SE 3e-12 |
| scanner_twfe (TWFE) | 0.58 s | 0.57 s | 1.01x | demean-bound |
| cs_cov40 (CS dr per-cell) | 4.40 s | 3.57 s | **1.23x** | dATT = 0 (bit-stable) |
| survey_absorb (WLS -> numpy twin) | 2.77 s | 2.59 s | 1.07x | dATT = 0 |
| skiprank_micro (2.4M×130 direct) | 6.88 s | 4.42 s | **1.56x** | dATT 1e-12 |

Memory: rust-side allocator high-water on the 2.4M×130 clustered solve
(alloc-profile build) drops **7.27 GB -> 2.66 GB** knob-on — the fast path's
working set is essentially the single equilibrated buffer, sidestepping the
SVD residual memory floor documented below (that floor still governs the
default path).

**Why opt-in and not default**: the default path's byte-stability is load-bearing
(benchmark identity conventions, golden pins at 1e-8) and the SVD truncation is
deliberate defense-in-depth. Within the opt-in path, parity is tol-bounded
(fitted ~1e-8 abs, SE ~1e-6 rel), never bit-level. A default-on flip is tracked
as a TODO.md follow-up (needs golden recapture + certification-rate telemetry +
the staged default-flip protocol).

---

## FE-absorption baseline: the MAP demeaning hot path (v3.6.x, July 2026)

A measured head-to-head against pyfixest 0.60 (Rust demeaner core), prompted by
Instacart's high-cardinality marketplace-modeling writeup, established that the
method-of-alternating-projections demeaning engine (`demean_by_groups`,
`diff_diff/utils.py`) is the dominant cost of every TWFE-family fit - and that
the gap is machinery, not architecture:

- **The engine is mainline, not edge.** `within_transform` is a thin two-way
  wrapper over `demean_by_groups`, so every fit of TwoWayFixedEffects,
  SunAbraham, BaconDecomposition, and WooldridgeDiD runs the MAP loop;
  DiD/MultiPeriodDiD hit it via `absorb=`; the survey replicate path re-runs
  the full weighted demean once per replicate (~80-200x per fit).
- **Attribution.** cProfile on SunAbraham's canonical staggered entry/exit
  panel (30k units x 30 periods, 540k rows): 13.5s fit with **82% inside
  `demean_by_groups`** - 2,948 `groupby().transform("mean")` calls, of which
  3.9s is pandas rebuilding the group index (`result_index_and_ids`) on every
  call. The demeaning share of fit time across the estimator sweep was 64-87%.
- **Machinery gap.** On identical 5M-row arrays at tol=1e-8, the pandas MAP
  loop is ~15x slower than pyfixest's compiled demeaner (2.39s vs 0.15s
  two-way; 3.01s vs 0.20s three-way). Estimator-level gaps are 2-7x on
  realistic workloads (shared fit costs dilute the demeaner share). Peak RSS
  is at parity or better - memory is a non-issue.
- **Iteration regimes** (two-way, tol=1e-8, measured): balanced panel 2
  iterations; 10% random gaps 6; entry/exit lifetimes at 60% span 16-17;
  order-level random incidence 8-9; contiguous 20% lifetimes 238-279 - the
  last crosses the `max_iter=100` cap, so the fit warns and returns
  slightly-off residuals where fixest/reghdfe/pyfixest (cap 10,000) converge.
- **Measured fix candidate.** A factorize-once + `np.bincount` rewrite of the
  MAP loop (pure numpy, identical convergence semantics) measured **3.4x** on
  the 5M-row two-way case and reaches full convergence on the stress case in
  19.2s where the current loop burns 25.1s hitting the cap without
  converging. The same pattern already exists in-tree
  (`diff_diff/survey.py` `_PsuScaffolding`).

The measurement surface is the seven-scenario FE-absorption suite
(`benchmarks/speed_review/bench_fe_absorption.py`; shapes documented in
`docs/performance-scenarios.md` scenarios 7-13; committed baselines
`baselines/fe_absorption_{before,after}.json`). The optional pyfixest
yardstick (`bench_fe_absorption_pyfixest.py`, guarded on import) asserts
coefficient parity < 1e-6 on the four exact-estimand scenarios so the
comparison stays honest over time.

**Landed (v3.6.x): factorize-once + bincount MAP rewrite.** End-to-end fit
speedups of 1.6-2.7x across the headline scenarios (table below). The
correlated-FE stress case now CONVERGES (~270 iterations under the raised
`max_iter=10_000` cap; ~2.7x cheaper per iteration) where the old loop hit
the cap of 100 and returned slightly-off residuals - its 0.8x wall-clock
ratio is the price of correctness: it runs 2.7x the iterations to full
convergence AND pays the exact LSMR span confirmation that now catches its
truncation-masked FE-spanned `post` column (previously kept as a junk
regressor in the design). The identity gate held at 1e-14-1e-16 on all
scenarios except the two carrying FE-spanned regressors through the DiD
`absorb=` path, whose estimates moved once (survey_absorb ATT by 9.8e-6;
geo_experiment SE by 1e-7) when the FE-spanned junk columns were snapped to
zero - a deliberate correction documented in REGISTRY "Absorbed Fixed
Effects" and the CHANGELOG. Remaining gap vs the pyfixest yardstick (2-26x
depending on scenario) is the compiled+parallel margin; a Rust demean kernel
in the existing `rust/` backend is the candidate next step, to be scoped
only if the remaining gap matters in practice (pyfixest itself validates
that architecture).

**Landed (v3.6.x): Rust `demean_map` kernel** (optional backend; numpy
canonical). Rayon-parallel MAP sweeps across the demeaned variables with the
GIL released; equivalence locked by iteration-count equality +
assert_allclose(atol=1e-12) against `_demean_map_numpy`. Measured on frozen
code against a fresh same-session numpy-backend run (medians, CVs <= 4.1%):
firm_churn 48.98s -> 25.84s (1.90x; CV-adjusted lower bound 1.78x - the
submit gate), tail_stress 31.74s -> 11.37s (2.79x), geo_experiment 0.976s ->
0.484s (2.02x, now ahead of the pyfixest yardstick's 0.623s), scanner 1.59x,
survey 1.44x, county 1.36x, guard_small unregressed. Identity vs the
committed after-baselines: 1e-13-1e-16 on every scenario incl. tail_stress.
**Solver-phase memory: final diagnosis + fix (2026-07, PR-D/PR-E).** The
fit-level peak on wide absorbed designs lives in the SOLVE phase on BOTH
backends. There is NO systematic rust-vs-python peak-RSS gap: an
interleaved A/B (3 rounds, alternating fresh subprocesses) measured rust
17.8 GB mean vs python 18.1 GB on firm_churn, each swinging +-1.5 GB
run-to-run - the historical "13.4 vs 21.0 GB backend gap" was a run-order
artifact (macOS's memory compressor evicts cold pages from the resident
set, so single-run `ru_maxrss` readings are machine-state lottery; gate
memory claims on allocation-level instruments or interleaved repeats,
never one RSS pair). The genuine waste was marshalling: the rust
`solve_ols` held ~6 concurrent n x k blocks (defensive input copy, scaled
clone, faer conversion copy, thin-SVD U, an ndarray COPY of U made only
for a k-vector dot, all scope-held under the vcov scores block) and the
python path formed a discarded Q in the rank-detection QR plus let gelsd
re-copy a C-order temporary. The PR-E slim (norms from the borrowed view,
equilibration fused into the single faer copy, U^T y off the faer factor,
early factor drops; python: `qr(mode="r")` + F-order `overwrite_a=True`
lstsq) cut the rust-side allocator high-water on the 2.4M x 130 clustered
solve from **15.32 GB to 7.81 GB** (6.14 -> 3.13 blocks, measured with the
feature-gated `alloc-profile` counting allocator) and, because the removed
copies also cost time, delivered CV-clear wall-clock wins on frozen-code
arms: firm_churn rust 25.84 -> 20.34 s (-21%), county 1.76 -> 1.55 s /
2.39 -> 2.09 s (rust/python, the skipped dorgqr), geo rust -6%, survey
rust -2%; python-arm estimates bit-identical (identity deltas exactly
0.0). Remaining floor is inherent to SVD-based lstsq (fused input copy +
thin-SVD U transient + vcov scores block + LAPACK gelsd internals);
further reduction = the tall-skinny-QR path parked with the QR-reuse row
in DEFERRED.md. `DIFF_DIFF_BACKEND=python` remains an escape hatch, and the
opt-in `DIFF_DIFF_DEMEAN_CHUNK_COLS` knob (PR-D) still caps the demean
dispatch transients.

### FE-absorption suite results

<!-- TABLE:start fe_absorption -->
| Scenario | n rows | Before (s) | After (s) | Speedup | Rust (s) | Rust speedup | pyfixest (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 7. County policy event study (SunAbraham) | 177,289 | 3.865 (cv 2.5%) | 2.087 (cv 2.5%) | 1.9x | 1.554 (cv 2.9%) | 1.3x | 0.190 (cv 4.0%, proxy) |
| 8. Firm panel with churn (SunAbraham) | 2,400,000 | 92.957 (cv 0.9%) | 46.127 (cv 2.7%) | 2.0x | 20.341 (cv 2.5%) | 2.3x | 1.912 (cv 20.3%, noisy, proxy) |
| 9. Scanner store-week (TWFE) | 3,255,000 | 1.549 (cv 0.3%) | 0.980 (cv 0.8%) | 1.6x | 0.612 (cv 0.3%) | 1.6x | 0.644 (cv 12.3%, noisy) |
| 10. Geo experiment 5M orders (DiD absorb) | 5,000,000 | 2.630 (cv 0.6%) | 0.942 (cv 0.0%) | 2.8x | 0.453 (cv 0.6%) | 2.1x | 0.623 (cv 7.8%) |
| 11. Survey BRR replicates (DiD absorb) | 500,000 | 7.385 (cv 8.6%) | 4.039 (cv 0.3%) | 1.8x | 2.778 (cv 0.1%) | 1.5x | - |
| 12. Correlated-FE stress (DiD absorb) | 5,000,000 | 26.271 (cv 0.4%) | 31.656 (cv 0.4%) | 0.8x | 11.381 (cv 0.3%) | 2.8x | 3.075 (cv 1.5%) |
| 13. Small-panel guard (TWFE) | 20,000 | 0.005 (cv 4.1%) | 0.004 (cv 4.9%) | 1.3x | 0.004 (cv 5.1%) | 1.0x | 0.007 (cv 2.2%) |

*noisy* = CV above the 10% unusable threshold from the noise protocol. *Rust speedup* is vs the After column (numpy engine, same code). *proxy* = timing-only pyfixest stand-in: the Sun-Abraham scenarios run a saturated `i(rel_time)` event study there (pyfixest 0.60 has no `sunab()`) - comparable demeaning load, different estimand, so those cells are not exact-estimand comparisons (see `bench_fe_absorption_pyfixest.py`).
<!-- TABLE:end fe_absorption -->

`tail_stress` is a deliberately adversarial correlated-FE shape and is
reported separately from the headline scenarios; the BEFORE baseline records
its non-convergence warning as evidence of the `max_iter=100` contract issue.

---

## CallawaySantAnna aggregation IF-assembly rewrite (v3.7, 2026-07)

Profiling at 2M+ rows showed `staggered_aggregation._compute_combined_influence_function`
was 56-85% of every analytical CS fit: called once per aggregation target (38 targets at
20 periods, 128 at 60 periods; plus 1 + N_e bootstrap-prep calls), each call re-ran
per-group full-DataFrame scans (`df[df["first_treat"] == g][unit].nunique()` — cost scales
with row count × frame width × cohorts × targets: 105 scans/fit at 5 cohorts, 975 at 15),
per-unit Python dict-lookup loops, and dense `(n_units × n_gt)` WIF matrices (~76MB × 3
arrays per call at 100k units). The rewrite: per-fit cohort tables (`np.unique` +
`np.bincount`, identity-validated cache) + closed-form WIF + fancy-index scatter; general
path preserved as fallback. Point estimates bit-identical; aggregated SEs ≤5e-16 relative.

Measured (medians of 3, M4 Max, Rust backend, otherwise-idle machine; BEFORE = pristine
`origin/main` worktree, AFTER = frozen branch):

| Scenario | Before | After | Speedup | Peak RSS |
|---|---:|---:|---:|---|
| no-cov analytical, 100k units × 20p (2M rows) | 1.32s | 0.21s | **6.3x** | 960→582 MB |
| no-cov analytical, 250k units (5M rows) | 3.57s | 0.60s | 5.9x | 2.1→1.3 GB |
| 5-cov reg / ipw / dr (2M rows) | 1.96/2.15/2.49s | 0.56/0.73/1.08s | 3.5/2.9/2.3x | ~-20% |
| 10/20/40-cov dr (2M rows) | 3.3/4.9/9.3s | 1.5/2.6/5.7s | 2.1/1.9/1.6x | 4.1→3.6 GB @40 |
| 40 periods × 10 cohorts (4M rows) | 9.09s | 3.91s | 2.3x | 2.6→2.1 GB |
| 60 periods × 15 cohorts (3M rows) | 9.86s | 4.34s | 2.3x | 2.5→1.9 GB |
| bootstrap-999 legs (2-4M rows) | 7.2-33.6s | 4.9-24.2s | 1.4-1.6x | ~-10-25% |
| RCS no-cov / dr / dr+boot (2M obs) | 4.2/7.3/44.0s | 1.5/4.4/39.5s | 2.8/1.7/1.1x | **6.4→2.1 / 7.0→3.2 GB** |

The RCS memory drop is the headline tail-risk fix: in RCS mode the WIF matrices were
observation-scale. Remaining CS bottlenecks after this rewrite (honest forecast, next
targets): `_precompute_structures` pandas groupby+pivot (now #1 on analytical no-cov
fits), the bootstrap draw-loop matmul (Phase 2 candidate; 999 × 100k × 390 cells),
and per-cell DR/IPW nuisance solvers at high covariate counts (IRLS `lstsq`, rank-guard
QR — Phase 3 candidate).

**R `did` yardstick** (did 2.5.1 — the 2.5.0 performance release with `faster_mode` and
the tiled-contraction bootstrap — R 4.5.2 + OpenBLAS 0.3.33; equal work: `att_gt` +
`aggte` simple+dynamic+group vs `fit(aggregate="all")`; identical `biters`/`cband`;
medians of 3; run configs recorded in each JSON artifact's metadata):

| Scenario | diff-diff | R 1-core | R 14-core (`pl=TRUE, cores=14`) |
|---|---:|---:|---:|
| no-cov analytical (2M rows) | 0.21s | 2.29s | 2.29s |
| 5-cov dr analytical | 1.09s | 3.67s | 3.74s |
| no-cov bootstrap 999 | 5.11s | 16.33s | 16.14s |
| 5-cov dr bootstrap 999 | 6.13s | 18.39s | 17.58s |
| 40p × 10c, 5-cov (4M rows) | 3.88s | 13.76s | 13.72s |

3-11x vs R at equal work under both R threading modes. The flat R multi-core column is
not a rigged comparison: did 2.5's bootstrap is a tiled BLAS contraction that already
multithreads through OpenBLAS, so fork-level `pl`/`cores` parallelism adds nothing on
this path — both arms are effectively multithreaded R. ATTs match R exactly at displayed
precision; analytical SEs to 5 significant digits; bootstrap SEs within ~1% (Monte Carlo).

---

## CS per-cell solver fast paths (Phase 3, 2026-07)

Fresh profile on main 49bbde68 (CallawaySantAnna dr, 100k units x 20 periods =
2M rows, 95 cells) after Phase 1 removed the aggregation dominance:

| item | cov10 (fit 1.56s) | cov20 (2.71s) | cov40 (5.68s) |
|---|---|---|---|
| `np.linalg.lstsq` in solve_logit IRLS (184 calls) | 0.39s | 0.81s | 1.99s |
| `_detect_rank_deficiency` pivoted QR (141 calls) | 0.15s | 0.31s | 0.84s |
| rust solve_ols SVD (out of scope, DEFERRED.md row) | 0.23s | 0.46s | 1.03s |

Two pure-Python fast paths in `diff_diff/linalg.py`:

1. **IRLS inner solve**: per-iteration gelsd SVD on the tall weighted design
   -> equilibrated normal equations + Cholesky, guarded by a LAPACK `dpocon`
   reciprocal-condition estimate (guard 1e-6; cho_factor alone can succeed
   with garbage on cond(G) ~1e10-1e16, and working weights can crush a
   column's effective scale on near-separated subgroups). Any uncertified
   iteration falls back to the exact legacy lstsq line. Columns are
   equilibrated ONCE (the repo removed a prior un-equilibrated cho_solve OR
   fast path for scale sensitivity - this addresses that removal reason);
   IRLS state stays in the raw basis (a scaled-basis tol would be ~sqrt(n)x
   tighter for every fit). Fallback count exposed via
   `diagnostics_out["irls_chol_fallback_iters"]`.
2. **Rank-detection stage-0 certification**: the always-run pivoted QR on the
   tall design is short-circuited by a Gram/eigvalsh certification on the
   equilibrated Gram at the `_rank_guarded_inv` 1e-10 threshold (two orders
   stricter than the QR boundary; never certifies what QR would call
   deficient); n < k, non-finite, looser-rcond, and uncertifiable designs
   fall through to the existing two-stage QR verbatim.

Measured arms (3-rep medians; BEFORE = pristine main 49bbde68 via
baseline-worktree PYTHONPATH, .so rebuilt at that SHA; BLAS threads pinned;
`.bench-local/cs_solver_arms_results.jsonl`):

| scenario (@100k x 20p) | rust BEFORE -> AFTER | speedup | python pair |
|---|---|---|---|
| cov10_dr | 1.53 -> 1.03s | 1.49x | 1.49x |
| cov20_dr | 2.64 -> 1.61s | 1.64x | 1.65x |
| cov40_dr | 5.69 -> 3.07s | 1.85x | 1.95x |
| cov20_ipw | 1.79 -> 0.90s | 1.98x | 2.03x |
| cov20_survey_dr | 2.73 -> 1.68s | 1.63x | 1.60x |

Component stages at cov40: solve_logit 2.54 -> 0.37s (6.8x),
_detect_rank_deficiency 0.84 -> 0.05s (16.5x). Instrumented during arms:
Cholesky fallback 0 iterations on every scenario/backend/arm; certification
rate 100% (141 -> 0 pivoted QRs per dr fit, 46 -> 0 per ipw fit);
rust-dispatch counts identical between arms (the solve_ols routing boolean is
provably unchanged). Deltas: overall ATT/SE exactly 0, per-cell max ~7e-15,
maxrss flat. After this the top per-cell item at cov40 is the rust solve_ols
SVD (1.04s; follow-up DEFERRED.md row), then pandas frame prep.

---

## CS multiplier-bootstrap fused tiled-GEMM rewrite (v3.7 Phase 2, 2026-07)

Stage-level instrumentation of `_run_multiplier_bootstrap` at the 40p × 10c × 100k-unit
flagship (390 cells, 72 event times, 999 draws; 20.9s of a 24.4s fit) split the loop as:

| stage | time | share |
|---|---:|---:|
| per-cell fancy-index copies `W[:, idx_t]`, `W[:, idx_c]` | 20.45s | **95%** |
| per-cell GEMVs | 0.73s | 3% |
| event-study GEMVs (72 full-width) | 0.27s | 1% |
| weight generation (rust, 1e8 draws) | 0.04s | 0.2% |

Not FLOPs, not RNG — ~100GB of cache-hostile gather traffic from slicing the
`(block × n_units)` weight matrix twice per cell (mean cell density 0.33: never-treated
controls appear in every cell). The fix: scatter every perturbation column (per-cell IFs,
overall combined IF, per-event-time combined IFs) into a column-tiled influence matrix
and run one BLAS GEMM per (weight block, column tile) — `bootstrap_chunking.tiled_if_matmul`
+ `ReplayableWeightStream` (each column tile replays the bit-identical weight stream via
RNG state snapshot/restore; byte-capped tiles keep the kernel's live intermediates bounded
regardless of n_units or column count). A scipy-sparse cell-column variant was measured and
rejected (27x slower than dense GEMM at 33% density). EfficientDiD's per-cell full-width
GEMV bootstrap routes through the same kernel via lazily materialized scaled-EIF columns.

Measured (5-rep medians, M4 Max, rust backend + Accelerate, otherwise-idle machine;
BEFORE = pristine `origin/main` worktree via PYTHONPATH, AFTER = frozen branch):

| Scenario | Before | After | Speedup | Peak RSS |
|---|---:|---:|---:|---|
| 40p × 10c, 5-cov dr, boot-999 (4M rows, 390 cells) | 24.49s | 4.28s | **5.7x** | 2.9→2.9 GB |
| no-cov boot-999 (2M rows) | 5.64s | 0.31s | **18.3x** | 2.0→1.6 GB |
| 5-cov dr boot-999 (2M rows) | 6.49s | 1.19s | **5.5x** | 2.3→1.9 GB |
| survey-PSU 5-cov dr boot-999 (500 PSUs, non-identity map) | 4.22s | 1.54s | **2.7x** | 2.0→1.9 GB |
| RCS 5-cov dr boot-999 (2M obs) | 42.15s | 40.07s | 1.1x | 3.3→3.2 GB |

ATT deltas exactly 0 on every scenario (the bootstrap never feeds point estimates);
bootstrap SE relative deltas 3e-16 to 1.2e-15 (BLAS reassociation only). The RCS scenario
barely moves because its fit time is the analytical repeated-cross-sections path, not the
bootstrap. EfficientDiD's bootstrap stage went 0.018s → 0.012s but is invisible inside its
fits: EfficientDiD's **analytical** stage has a pre-existing pathological hot loop
(339s at just 2k units × 20p with 5 covariates; sampled to a broadcasted subtract/multiply
in a Python-level loop) — tracked in TODO.md as its own item, out of scope here.

Remaining CS bottlenecks (updated forecast): `_precompute_structures` pandas
groupby+pivot on analytical no-cov fits, and per-cell DR/IPW nuisance solvers at high
covariate counts (IRLS `lstsq`, rank-guard QR — Phase 3 candidate).

---

## EfficientDiD analytical path: omega_ridge + fused tiled GEMM (v3.7, 2026-07)

Resolves the "analytical stage pathological scaling" TODO filed during the Phase 2
bootstrap work. Instrumented stage split at n=500/1k/2k units (20 periods, 5 cohorts,
5 covariates, 95 cells, H=60 pairs/cell; `.bench-local/edid_analytical_*.json`):

| stage | n=500 | n=1k | n=2k | mechanism |
|---|---|---|---|---|
| `compute_omega_star_conditional` | 23.9s | 71.3s | 289.8s | O(n^2 H^2) Python pair loop, ~4 (n x n_group) broadcast temporaries per pair |
| `compute_per_unit_weights` | 11.9s | 23.9s | 47.7s | per-unit SVD cond + SVD pinv Python loop |
| everything else | ~0.4s | ~0.5s | ~1.8s | sieves, gen_out, EIF |

The spike also uncovered the numerical-stability story documented in REGISTRY.md
(Omega* ridge Note): Omega* is numerically singular by construction (cond 1e17-1e22,
100% of units), so the legacy pseudoinverse redrew per-cell values at ~1e-2 rel under
ANY floating-point change (1-ulp input perturbation on the shipped code: 1.2e-4).
An eigh-based pinv (2x faster) was measured and REJECTED — O(1) per-unit weight
changes through the same cutoff cliff. The ridge (default 1e-6, trace-scaled) is both
the stability fix and what unlocks the batched solve.

Rewrite (one PR): `_kcov_batch` GEMM identity `KCov = W@(A0*B0) - (W@A0)*(W@B0)`
(rows of the kernel matrix sum to 1, survey-weighted included; validated 28-29x per
cell at 2e-15 rel on captured inputs), (t_pre_j, t_pre_k) dedup (1830 pairs -> ~190
GEMM columns), fused unit-tiled pass 2 (per-group kernel matrices computed once per
tile, reused across ALL 95 cells; omega assembled per tile; batched
`np.linalg.solve` ridge weights; EIF centered after all tiles;
`_TARGET_OMEGA_TILE_BYTES` = 256MB, call-time resolved), nocov Gram twin.

Measured arms (3-rep medians; BEFORE = pristine main ced49e89 via baseline-worktree
PYTHONPATH; `.bench-local/edid_arms_results.jsonl`):

| scenario | BEFORE | AFTER | speedup | maxrss B->A |
|---|---|---|---|---|
| conditional n=500 | 35.9s | 1.83s | 19.7x | 0.18 -> 0.25 GB |
| conditional n=1k | 95.4s | 3.48s | 27.5x | 0.25 -> 0.37 GB |
| conditional n=2k | 337.3s | 7.81s | 43.2x | 0.37 -> 0.59 GB |
| conditional n=10k | ~2.3h (extrapolated O(n^2)) | 57.7s | ~146x | - -> 0.71 GB |
| conditional n=100k | memory-killed | 85.6 min (1 rep) | completes | - -> 4.0 GB |
| nocov n=2k | 1.22s | 0.47s | 2.6x | 0.15 -> 0.15 GB |
| survey n=1k | 95.3s | 3.34s | 28.6x | 0.25 -> 0.37 GB |

Estimate deltas AFTER vs BEFORE (the documented one-time ridge redraw, shrinking
with n as the indeterminacy band tightens): overall ATT 1.6e-2 / 4.0e-3 / 1.1e-3 rel
at n=500/1k/2k; worst post-treatment cell <= 0.58 of its own SE; nocov path ~1e-7.
1-ulp per-cell stability after: <= 3e-9 rel (before: 1.2e-4).

The lever identified at the time — cross-cell kernel-table hoisting (Term 5
tables are (g,t)-independent; Term 2 depends only on t), estimated ~10x on the
intrinsic per-cell GEMM factor at 100k-unit scale — was filed as a TODO row and
is RESOLVED by the follow-up below.

### Cross-cell kcov-table hoisting + per-group W lifecycle (2026-07 follow-up)

Resolves the hoisting TODO row above. Instrumented stage split of the v3.7 tiled
path (`.bench-local/edid_hoist_spike.py`; same 20p/5-cohort/5-cov scenario):

| stage | n=2k | n=10k | n=30k | scaling |
|---|---|---|---|---|
| t5 + t2 kcov GEMMs | 0.95s | 21.2s | 217.2s | O(n^2), dominant |
| ridge_solve | 3.5s | 16.9s | 48.5s | linear |
| assembly (Python (j,k) loop) | 1.4s | 6.6s | 28.3s | linear x tile-count overhead |
| fit total | 7.7s | 57.2s | 359s | |

Duplication: t5 computed 108,300 kcov columns per tile where 1,140 are distinct
(95x); t2 18,050 vs 3,610 (5x) — ~26.6x combined GEMM-traffic reduction. A
"base-table" bilinear factorization (level covariances combined 4-term, ~90x)
was measured and REJECTED: ~500x fp-error amplification on persistent panels
(`.bench-local/edid_hoist_cancellation.py`), which through the ridge's 1/lambda
sensitivity cap lands AT the 1e-6 stability contract. The shipped variant hoists
the DIFFERENCED-column tables (numerics identical in kind to the per-cell
construction). Assembly variants were micro-benchmarked: full (H,H) index
gathers were memory-bandwidth-bound and SLOWER than the legacy Python loop;
the shipped triu-strip gather through compact per-cell column slices is ~5x
faster than the naive vectorization and faster than the loop, while preserving
the legacy per-entry operation order value-exactly (term 1 folded into the
compact slice pre-gather; mirror preserved at the copy level). Kernel W
matrices are built and freed one group at a time, so the tile budget is set by
the largest single group (~3-5x fatter tiles at 100k).

Measured arms (3-rep medians; BEFORE = pristine main fcd77683 via
baseline-worktree PYTHONPATH; `.bench-local/edid_hoist_arms_results.jsonl`):

| scenario | BEFORE | AFTER | speedup | maxrss B->A |
|---|---|---|---|---|
| conditional n=2k | 7.71s | 6.96s | 1.11x | 0.59 -> 0.59 GB |
| conditional n=10k | 57.2s | 35.7s | 1.60x | 0.76 -> 0.79 GB |
| conditional n=30k | 358.4s | 128.8s | 2.78x | 0.96 -> 1.12 GB |
| conditional n=100k | 85.6 min (v3.7 measured, 1 rep) | 17.8 min (1 rep) | ~4.8x | 4.0 -> 4.02 GB |
| nocov n=2k (guard) | 0.44s | 0.44s | 1.00x (byte-identical) | flat |
| survey n=1k | 3.30s | 3.03s | 1.09x | flat |

The kernel-covariance stage itself: 21.2s -> 0.53s at 10k (40x). Deltas:
post-treatment cells ~1e-12 rel (max 6e-12), overall ATT ~1e-13, nocov exactly
0. Post-hoist the top stages at every scale are `_ridge_solve_weights` (batched
LAPACK LU over (n x H x H) stacks; 17.2s at 10k — Rust batched-Cholesky is the
natural follow-up, resolved by the section below) and the nuisance sieves.

### Rust batched-Cholesky ridge solve (2026-07 follow-up)

Resolves the ridge-solve TODO row above. `_ridge_solve_weights`' batched
`np.linalg.solve` is a LAPACK dgesv gufunc — LU (2/3 H^3 flops) and SERIAL over
the batch — although the ridged Omega* is SPD (Cholesky = 1/3 H^3). Spike on
real captured batches (`.bench-local/edid_ridge_solve_spike.py`, cond_10k: 570
calls, all H=59/60, m~1300-1740): numpy stage 15.6s extrapolated vs Rust kernel
1.6s = **9.6x kernel aggregate**; zero non-PD rows (min relative eigenvalue
1.7e-8 = the ridge floor), so the LU/NaN-poison fallback is pure defense. Raw
solve divergence Cholesky-vs-LU: max ~3.7e-7 per entry (cond ~6e7 x eps class)
— but fit-level deltas land at ~1e-12 because the weight perturbations live in
Omega*'s near-null directions (near-duplicate moments with near-identical
generated outcomes), exactly the insensitivity the ridge Note's "any fixed
weights summing to 1" argument predicts.

Kernel-variant A/B (serial, per-H, m=1737): hand-rolled unblocked Cholesky in a
reused per-thread scratch vs faer `Llt::new` (which allocates a fresh Mat +
scratch per call): hand-rolled wins 10.3x at H=2 and 1.9x at H=8; faer wins
~1.4x at H=60 (SIMD factorization). Shipped hand-rolled (user-confirmed): zero
per-call allocation, wins where H is small, one oracle-locked code path; the
fit-level cost of not switching at H=60 is ~2% (the win is rayon parallelism
across the batch — Accelerate's serial dgesv is roughly at parity with the
serial hand-rolled Cholesky per solve).

Measured arms (3-rep medians; BEFORE = pristine main 81d53f59 via
baseline-worktree PYTHONPATH; `.bench-local/edid_chol_arms_results.jsonl`):

| scenario | BEFORE | AFTER | speedup | maxrss B->A |
|---|---|---|---|---|
| conditional n=2k | 7.08s | 4.29s | 1.65x | 0.59 -> 0.60 GB |
| conditional n=10k | 36.2s | 22.7s | 1.60x | 0.80 -> 0.75 GB |
| conditional n=30k | 129.0s | 90.1s | 1.43x | 1.09 -> 1.13 GB |
| conditional n=100k | 17.76 min (#629 measured, 1 rep) | 15.92 min (1 rep) | 1.12x | 4.02 -> 3.84 GB |
| nocov n=2k (guard) | 0.44s | 0.45s | 1.0x (path untouched) | flat |
| survey n=1k | 3.03s | 1.68s | 1.80x | flat |
| pure-python cond n=2k (guard) | 6.09s | 6.10s | byte-identical results | flat |

Ridge stage at 10k: 17.1s -> 3.68s (4.65x; the aspirational >=5x stage gate was
narrowly missed — the kernel is 9.6x but ~2s of the remaining stage is shared
Python prep: the zero_mask full-stack abs scan + the `omega_stack[rest]`
fancy-index copy, ~28 GB extra memory traffic per 10k fit; TODO row filed for
the copy shave). Deltas: post cells max ~2e-12 rel, overall ATT ~1e-13,
event-study max ~5e-10, all cells incl. near-zero placebos <= 1.4e-11 abs;
pure-python arm byte-identical. After this the largest O(n) stage is the
sieve/nuisance construction outside the tiled pass (~9s at 10k).

---

## Memory scaling at the millions-of-units tail (v3.x, June 2026)

At the scale where the dense working arrays - not compute - are the binding constraint
(millions of units, few periods), three 2026-06 refactors cut peak resident memory for
the multiplier-bootstrap and within-transform paths:

- **CallawaySantAnna / EfficientDiD / HeterogeneousAdoptionDiD multiplier bootstraps**
  (#561, #563) generate and consume the dense `(n_bootstrap x n_units)` weight matrix one
  draw-block at a time via `diff_diff/bootstrap_chunking.py`, instead of materializing it
  in full (`999 x 5,000,000 x 8` bytes is ~40 GB).
- **`within_transform`** (#567) drops a redundant defensive full-frame copy and attaches the
  demeaned columns via a single `pd.concat` (under copy-on-write the originals are shared,
  not copied).

Numerical contract: the `within_transform` change (#567) is **bit-identical** to the prior
code - it only changes frame assembly, so estimates match at `atol=0`. For the bootstrap
chunking (#561, #563) the generated weight *stream* is bit-identical, but the downstream
`weights @ influence` statistics match the un-chunked path only to within BLAS floating-point
reassociation (~1 ULP, far below bootstrap Monte-Carlo error) - not bit-for-bit. Either way
the working arrays are the same size, so these **peak-memory** numbers are unaffected.

Measured with `benchmarks/speed_review/bench_memory_scaling.py` - each config fit in an
isolated subprocess, peak = `resource.getrusage(...).ru_maxrss`, median of repeated runs,
`--backend python` on Apple Silicon. "Before" is the pre-#561 tree; "after" is current `main`:

| Config (999 bootstrap reps where applicable) | Before | After | Reduction |
|---|---:|---:|---:|
| CallawaySantAnna bootstrap, 500k units | 12.9 GB | 2.1 GB | **-84%** |
| CallawaySantAnna bootstrap, 1M units | 13.5 GB | 3.0 GB | **-78%** |
| EfficientDiD bootstrap, 500k units | 8.3 GB | 1.6 GB | **-81%** |
| HeterogeneousAdoptionDiD event-study cband, 500k units | 7.7 GB | 1.2 GB | **-84%** |
| TwoWayFixedEffects (hc1) fit, 500k units x 6 cov | 1.0 GB | 0.93 GB | -8% |
| TwoWayFixedEffects (hc1) fit, 1M units x 6 cov | 1.7 GB | 1.5 GB | -9% |

The bootstrap chunking removes the dominant allocation, so the previously out-of-reach
millions-of-units x 999-rep regime now stays near the fit's memory floor; the within-transform
copy elision is a smaller, broadly-applicable win on the TWFE-family fit path. Single-run
`ru_maxrss` on these transient-heavy paths is allocator-dependent (the un-chunked 1M run varied
~12.7-15.0 GB across repeats), so the table reports medians - the reduction ratios are stable.

Deferred (see `DEFERRED.md`): tiling the *stratified* survey-PSU weight generator (few PSUs, rarely
OOMs). The `ImputationDiD` / `TwoStageDiD` `_iterative_demean` vectorization deferred here
shipped later as the demean-modernization PR (both estimators now route through the shared
`demean_by_groups` MAP engine and the shared `_iterative_fe_solve` bincount FE solver).

---

## Practitioner Workflow Baseline (v3.1.3, April 2026)

Earlier sections of this document (v1.4.0, v2.0.3) measured isolated `fit()`
calls on synthetic panels for R-parity. This section measures **end-to-end
practitioner chains** - Bacon decomposition, fit, event-study pre-trend
inspection, HonestDiD sensitivity grids, cross-estimator robustness refits,
and reporting - at data shapes anchored to applied-econ papers and industry
writeups. The six scenarios are defined in
[`docs/performance-scenarios.md`](performance-scenarios.md); scripts live in
`benchmarks/speed_review/bench_*.py`; raw results in
`benchmarks/speed_review/baselines/*.json` and flame profiles in
`benchmarks/speed_review/baselines/profiles/`.

Environment: macOS darwin 25.3 on Apple Silicon M4, Python 3.9,
numpy 2.x, diff_diff 3.1.3. Each multi-scale scenario runs at three data
scales under both `DIFF_DIFF_BACKEND=python` and `DIFF_DIFF_BACKEND=rust`,
with one intentional exception: the SDiD few-markets scenario at its
`large` scale runs Rust only, because the pure-numpy jackknife at n=500
would exceed four minutes per run without changing the already-clear
Python-vs-Rust conclusion established at `small` and `medium`. The
numerical tables below are auto-generated from the committed JSON
baselines by `benchmarks/speed_review/gen_findings_tables.py`; narrative
prose is hand-written and must be re-read when numbers shift.

### Scale sweep - end-to-end wall-clock

Four of the six scenarios run at three scales (small / medium / large). The
small scale matches tutorial data shapes; medium reflects typical
practitioner workloads; large stretches toward the upper end of what an
analyst might bring (1M-row BRFSS microdata, 1,500-unit county-level
staggered panel, 1,000-unit multi-region brand survey, 500-unit zip-level
geo-experiment). Dose-response and reversible-dCDH run at a single mid-range
scale. Data-shape details are in `docs/performance-scenarios.md`.

<!-- TABLE:start scale_sweep_totals -->
| Scenario | Scale | Python (s) | Rust (s) | Py/Rust |
|---|---|---:|---:|---:|
| 1. Staggered campaign | small | 0.52 | 0.51 | 1.0x |
|  | medium | 0.81 | 0.81 | 1.0x |
|  | large | 1.32 | 1.31 | 1.0x |
| 2. Brand awareness survey | small | 0.23 | 0.20 | 1.1x |
|  | medium | 0.53 | 0.50 | 1.1x |
|  | large | 0.87 | 0.93 | 0.9x |
| 3. BRFSS microdata -> CS panel | small | 0.21 | 0.17 | 1.3x |
|  | medium | 0.49 | 0.47 | 1.0x |
|  | large | 1.33 | 1.32 | 1.0x |
| 4. SDiD few markets | small | 3.70 | 0.04 | 88.6x |
|  | medium | 4.00 | 0.11 | 37.6x |
|  | large | skip | 0.23 | - |
| 5. Reversible dCDH | single | 0.79 | 0.78 | 1.0x |
| 6. Pricing dose-response | single | 0.59 | 0.63 | 0.9x |
<!-- TABLE:end scale_sweep_totals -->

### Scaling findings

**Three findings are load-bearing for the optimization priority list:**

1. **BRFSS `aggregate_survey` is now practitioner-fast at every measured
   scale.** Prior to the precompute-scaffolding fix (see "Optimization
   landed" below), the full chain at 1M rows took ~24 seconds and was
   essentially all inside `_compute_stratified_psu_meat`. After the fix,
   the chain is sub-2s at every measured scale; `aggregate_survey`
   continues to dominate its own (now-cheap) chain share, but in
   absolute time the entire workflow is well under a practitioner-
   perceptible threshold at realistic pooled-multi-year BRFSS volume.
   The path is entirely Python, so Python and Rust backends track each
   other within noise.
2. **Staggered CS chain stays cheap across scales.** A 10x unit increase
   (150 -> 1,500) is a small-single-digit multiplier on total time.
   ImputationDiD and SunAbraham together consistently account for
   ~70-80% of the chain; either can be the single top phase at a given
   (scale, backend) cell, which is a per-cell ranking detail not a
   stable pattern to optimize against.
3. **SDiD Rust gap is stable across scales, not emergent.** Python SDiD
   has a fixed per-jackknife-refit overhead that dominates even at small
   n. Rust stays sub-second through 500 units.

**Two findings hold across scales:**

4. Brand-awareness survey total scales roughly linearly in n_units, but
   the JK1 replicate path inside it scales closer to
   n_units x n_replicates - faster growth than the chain total, so it
   increasingly dominates at large n.
5. Rust backend gives large uplift only for SDiD (order-of-magnitude
   and up). Elsewhere the gap is modest across all measured (scenario,
   scale) cells - see the scale-sweep table for exact ratios. The
   primary bottlenecks live in Python code the Rust backend does not
   touch (`aggregate_survey`, JK1 replicate fit), and paths that Rust
   does touch (CS bootstrap, ImputationDiD, Survey TSL) are already
   well-vectorized in Python.

### Top phases by scenario at largest measured scale

<!-- TABLE:start top_phases_by_scenario -->
| Scenario | Scale | Backend | Top phase (%) | 2nd phase (%) | 3rd phase (%) |
|---|---|---|---|---|---|
| 1. Staggered campaign | large | python | `6_imputation_did_robustness` (54%) | `5_sun_abraham_robustness` (21%) | `2_cs_fit_with_covariates_bootstrap999` (13%) |
| 1. Staggered campaign | large | rust | `6_imputation_did_robustness` (41%) | `5_sun_abraham_robustness` (36%) | `2_cs_fit_with_covariates_bootstrap999` (12%) |
| 2. Brand awareness survey | large | python | `3_replicate_weights_jk1` (46%) | `4_multi_outcome_loop_3_metrics` (26%) | `7_event_study_plus_honest_did` (17%) |
| 2. Brand awareness survey | large | rust | `3_replicate_weights_jk1` (50%) | `4_multi_outcome_loop_3_metrics` (25%) | `7_event_study_plus_honest_did` (15%) |
| 3. BRFSS microdata -> CS panel | large | python | `1_aggregate_survey_microdata_to_panel` (91%) | `5_sun_abraham_robustness` (8%) | `2_cs_fit_with_stage2_survey_design` (1%) |
| 3. BRFSS microdata -> CS panel | large | rust | `1_aggregate_survey_microdata_to_panel` (95%) | `5_sun_abraham_robustness` (4%) | `2_cs_fit_with_stage2_survey_design` (1%) |
| 4. SDiD few markets | medium | python | `5_sensitivity_to_zeta_omega` (43%) | `3_in_time_placebo` (39%) | `2_sdid_bootstrap_variance_200` (9%) |
| 4. SDiD few markets | large | rust | `5_sensitivity_to_zeta_omega` (38%) | `3_in_time_placebo` (30%) | `1_sdid_jackknife_variance` (16%) |
| 5. Reversible dCDH | single | python | `4_heterogeneity_refit` (51%) | `1_dcdh_fit_Lmax3_survey_TSL` (49%) | `3_honest_did_on_placebo` (0%) |
| 5. Reversible dCDH | single | rust | `4_heterogeneity_refit` (50%) | `1_dcdh_fit_Lmax3_survey_TSL` (50%) | `3_honest_did_on_placebo` (0%) |
| 6. Pricing dose-response | single | python | `1_cdid_cubic_spline_bootstrap199` (26%) | `6_spline_sensitivity_num_knots2` (25%) | `3_cdid_event_study_pretrend` (25%) |
| 6. Pricing dose-response | single | rust | `1_cdid_cubic_spline_bootstrap199` (26%) | `6_spline_sensitivity_num_knots2` (25%) | `3_cdid_event_study_pretrend` (25%) |
<!-- TABLE:end top_phases_by_scenario -->

Per-scenario phase narrative (cross-check against the table above after
any rerun):

- **Staggered campaign.** ImputationDiD robustness and SunAbraham
  consistently account for ~70-80% of the chain at every scale. They
  sit in a narrow phase-share band (each typically ~25-50%) and which
  one leads varies by (scale, backend) and can flip across reruns at
  medium scale where the two are close; see the table for the exact
  ordering per cell. CS fit with `n_bootstrap=999` (both with and
  without covariates) is well-vectorized and sits well below both in
  the ranking. Either phase is a legitimate optimization target; the
  aggregate share is what drives the "next hotspot" priority.
- **Brand awareness survey.** At small scale HonestDiD dominates. From
  medium onwards JK1 is the single largest phase under both backends;
  see the table for the exact share per cell. Python and Rust totals
  stay close across the sweep (within ~1.1x at any measured scale,
  see scale-sweep table); the JK1 replicate-fit loop is not
  Rust-accelerated, so the backends neither help nor hurt each other
  meaningfully on this chain.
- **BRFSS.** `aggregate_survey` remains the single largest chain share
  under both backends at every scale, but the absolute chain total is
  sub-2s at 1M rows after the precompute-scaffolding fix. Downstream
  phases (CS fit, SunAbraham, HonestDiD) are a fraction of a second
  combined - see the scale-sweep table for the current totals.
- **SDiD few markets.** `sensitivity_to_zeta_omega` and
  `in_time_placebo` are the two largest phases under Python at every
  scale and under Rust at medium/large (together ~70% of the chain).
  At Rust small the absolute cost collapses so far that per-phase
  fixed overhead dominates and `2_sdid_bootstrap_variance_200` slightly
  edges the other two. The difference across backends is absolute:
  under Python these phases drive a multi-second chain, under Rust
  they stay in the top ranks but of a sub-second total runtime. That
  is the Python-vs-Rust story for this scenario.
- **Reversible dCDH.** Main fit and heterogeneity refit are the two
  largest phases by design - together effectively the whole chain,
  with the remainder on HonestDiD at <2%. The two phases sit within a
  few percentage points of each other at this shape and the leader
  can flip across reruns under either backend. Both fits run under
  the same `SurveyDesign` and rebuild shared TSL scaffolding - that
  is the optimization opportunity, independent of which side is
  slightly larger on a given measurement.
- **Pricing dose-response.** Four spline fits account for essentially all
  runtime; linear scaling in variant count.

### Top hotspots ranked by total-time contribution

| # | Location | Scenario + scale | Signal | Recommended action |
|---|---|---|---|---|
| 1 | `diff_diff/survey.py` `_compute_stratified_psu_meat` + `aggregate_survey` | BRFSS @ 1M rows | previously dominated BRFSS chain at all scales (~100% at 1M rows) | **LANDED** (this PR). Precompute stratum-PSU scaffolding once per design at `aggregate_survey` top level; replace per-cell pandas groupby with two vectorized `np.bincount` passes. BRFSS-large chain drops from ~24s to sub-2s across both backends. See "Optimization landed" below. |
| 2 | `diff_diff/imputation.py` ImputationDiD fit (+ `diff_diff/sun_abraham.py` SunAbraham fit) | Staggered CS @ 1,500 units | together consistently ~70-80% of the chain at every scale; either can be the top phase at a given (scale, backend) cell | **Investigate only after BRFSS fix lands.** Total chain is well under practitioner-perceptible threshold; candidate follow-up. Either phase is a legitimate target. |
| 3 | `diff_diff/utils.py:1434` `_sc_weight_fw_numpy` | SDiD python @ any scale | dominates Python SDiD at all scales | **Already ported to Rust.** Python fallback acceptable as a teaching/safety path; non-production for n > 100. Python skipped at n=500 (jackknife cost would exceed 4 minutes per run). |
| 4 | `diff_diff/chaisemartin_dhaultfoeuille.py` dCDH fit + heterogeneity | Reversible (single scale) | main fit and survey-aware heterogeneity refit each rebuild TSL scaffolding; heterogeneity phase is as expensive as the main fit | **Cache/precompute** - heterogeneity refit duplicates the main fit's TSL setup under the same `SurveyDesign`. Not P0; newer code path (v3.1) never optimization-reviewed. |
| 5 | `diff_diff/continuous_did.py` CDiD spline bootstrap | Dose-response (single scale) | four spline fits ~equal, linear in variant count | **Leave alone** - well under perceptible threshold. |

### Memory analysis

End-to-end peak RSS and per-scenario growth are captured in each JSON
baseline under the `memory` field, recorded via a psutil background
sampler at 10 ms. A standalone `tracemalloc`-based allocator attribution
pass for the BRFSS-1M scenario lives at
`benchmarks/speed_review/mem_profile_brfss.py`; its scrubbed output is
in `benchmarks/speed_review/baselines/mem_profile_brfss_large_<backend>.txt`.

<!-- TABLE:start memory_by_scenario -->
| Scenario | Scale | Py peak RSS (MB) | Py growth (MB) | Rust peak RSS (MB) | Rust growth (MB) |
|---|---|---:|---:|---:|---:|
| 1. Staggered campaign | small | 146 | 31 | 148 | 34 |
|  | medium | 235 | 85 | 253 | 100 |
|  | large | 486 | 251 | 582 | 327 |
| 2. Brand awareness survey | small | 130 | 15 | 128 | 13 |
|  | medium | 183 | 45 | 189 | 55 |
|  | large | 340 | 139 | 348 | 158 |
| 3. BRFSS microdata -> CS panel | small | 133 | 11 | 130 | 8 |
|  | medium | 203 | 17 | 200 | 21 |
|  | large | 413 | 25 | 409 | 25 |
| 4. SDiD few markets | small | 124 | 10 | 116 | 1 |
|  | medium | 148 | 8 | 117 | 0 |
|  | large | skip | skip | 118 | 0 |
| 5. Reversible dCDH | single | 134 | 20 | 134 | 20 |
| 6. Pricing dose-response | single | 122 | 8 | 123 | 9 |
<!-- TABLE:end memory_by_scenario -->

The ~115-130 MB floor is the Python + diff-diff + numpy import footprint;
the "growth" columns are the practitioner-meaningful numbers.

### Memory findings

1. **BRFSS `aggregate_survey` was compute-bound, not memory-bound - and
   the compute side is now addressed.** Working-memory growth stayed in
   the low tens of MB across the 20x data-growth sweep (50K -> 1M rows);
   the pre-fix tracemalloc pass confirmed net retained allocation under
   1 MB and identified `tracemalloc`'s own linecache overhead as the
   top allocation site (smoking gun that nothing else was allocating
   meaningfully). The precompute-scaffolding fix in this PR is a pure
   CPU win - no change to the function's memory profile, which was
   already Lambda-friendly.
2. **Staggered CS chain is memory-heavier than wall-clock suggested.** At
   1,500 units the chain's peak RSS sits in the high-400s to high-500s
   MB depending on backend. Fine for workstations, tight for 512 MB
   Lambda tier. Bootstrap-999 in CS and ImputationDiD's saturated
   regression are plausible drivers. Rust uses slightly more memory here
   (likely FFI-held temporary array copies); not worth optimizing.
3. **JK1 replicate path is allocation-heavy at large replicate count.**
   At 1,000 units × 160 replicates the chain's growth during run sits in
   the mid-100s of MB (see memory table). Each replicate refit plus the
   n × n_replicates weight matrix drives this. A Rust port would save
   memory even though time is within noise today - the dual benefit
   strengthens the case for the port if replicate counts grow.
4. **SDiD Rust path is essentially memory-free** (growth at or below a
   single MB across scales). Rust does the work in native memory without
   round-tripping through the Python allocator. Confirms the existing
   Rust port is well-behaved on both axes.
5. **No scenario hits OOM territory at measured scales.** Peak RSS across
   the whole sweep stays under 600 MB. 1 GB is a comfortable ceiling for
   every scenario measured.

### Priority of optimization opportunities

| # | Opportunity | Time upside | Memory upside | Risk | Priority |
|---|---|---|---|---|---|
| 1 | `aggregate_survey` precompute stratum scaffolding | ~-20s at 1M rows | none (already memory-efficient) | Low | **LANDED** (this PR) |
| 2 | Staggered CS chain working-memory audit (Lambda-oriented) | none | ~200-300 MB at 1,500 units (peak RSS crosses 512 MB Lambda line under Rust) | Medium | Low (bump to Medium if Lambda deployment becomes a concrete ask) |
| 3 | dCDH: cache TSL scaffolding across main fit + heterogeneity refit | ~0.2s per chain | ~20 MB per chain | Low | Low |
| 4 | ImputationDiD fit-loop vectorization audit | ~0.1-0.3s at 1,500 units | unknown | Low | Low |
| 5 | Rust-port JK1 replicate fit loop | ~0.5s at 160 replicates | ~140 MB at 160 replicates | Medium | Low (demoted: Rust is no longer slower than Python on this path after rerun, so the "fix-a-Rust-regression" leg of the original rationale is gone) |

### Optimization landed

**#1 shipped in this PR.** `diff_diff/survey.py` now precomputes a
per-design `_PsuScaffolding` (strata codes, global PSU codes, per-
stratum counts and FPC ratios, singleton mask, lonely-PSU-aware
variance-computable flag).  `aggregate_survey` builds it once per call
and threads it through `_cell_mean_variance` so each per-cell variance
reduction uses two vectorized `np.bincount` passes instead of a
per-stratum pandas groupby loop.  Numerics are preserved to sub-ULP
tolerance; equivalence tests across seven design cases
(`TestAggregateSurveyScaffolding`) enforce `assert_allclose(atol=1e-14,
rtol=1e-14)` between fast and legacy paths.

Replicate-weight designs (JK1 etc.) continue to use the legacy
`compute_replicate_if_variance` code path and are unaffected.

**Bottom line: no practitioner-perceptible bottleneck remains in the
six measured workflows; four optional items stand by.** Items #2-5
above should be prioritized by concrete deployment-environment signal
(Lambda OOMs, practitioner
reports of slowness at specific shapes), not proactively.

### Correctness-adjacent observations (not P0, route separately)

These are developer-ergonomics / API-consistency smells surfaced during
scenario development. None are silent-failures and none belong in this PR
or in the silent-failures audit; logging here for awareness.

1. **`aggregate` / `level` parameter naming is inconsistent.** CS accepts
   `aggregate="event_study"`; ContinuousDiD requires
   `aggregate="eventstudy"` on `fit()` **but** `level="event_study"` on
   `to_dataframe()`. Two different spellings within one estimator plus a
   third cross-estimator spelling. Surfaced when the P1 exit-propagation
   fix stopped silently swallowing the resulting `ValueError` in the
   dose-response benchmark. Route: API-consistency cleanup, minor.
2. **`generate_survey_did_data(panel=True)` `treated` column.** Row-level
   active-treatment indicator that is zero in pre-periods, which makes it
   quietly incompatible with `check_parallel_trends` (expects unit-level
   treatment group membership) and pre-period placebo tests. Tutorial 17
   does not hit this because it uses a 2x2 design where `post` discriminates
   the comparison. Suggest adding a `treat_unit` column alongside `treated`
   for generator output clarity. Route: DGP cleanup, minor.
3. **`SurveyDesign.replicate_method` case sensitivity.** `"jk1"` raises
   `ValueError("must be one of {'Fay', 'SDR', 'BRR', 'JKn', 'JK1'}")`;
   `"JK1"` works. Either normalize the input or mention the expected casing
   in the error message. Route: API-ergonomics, minor.

### What this baseline does not answer

- OOM behaviour at the edge: the sweep captures peak RSS up to ~600 MB
  (staggered CS large under Rust). Behaviour under a hard memory ceiling
  (512 MB Lambda, 1 GB container) is not exercised; if deployment signal
  emerges that practitioners hit those ceilings, a ceiling-test pass
  should be added.
- Pure-Rust profiles: scenarios run the Rust backend as a black box.
  Optimizing inside `rust/` is a separate concern owned by the crate
  maintainers and is not in scope here.
- Real-data shapes: the scenarios use synthetic DGPs. The BRFSS scenario
  uses a BRFSS-shaped synthetic panel, not actual BRFSS microdata. If a
  real-data calibration becomes relevant, CDC BRFSS annual files are
  public.

### Reproducing

```bash
pip install pyinstrument                  # one-time, dev-only
python benchmarks/speed_review/run_all.py # both backends, all scenarios

# Single scenario, single backend:
DIFF_DIFF_BACKEND=rust python benchmarks/speed_review/bench_campaign_staggered.py
```

Raw JSON is written under `benchmarks/speed_review/baselines/` for
scenario-level diffing as the library evolves; flame HTMLs are written
alongside under `baselines/profiles/` (gitignored; regenerated on each run).

---

## Results Achieved (v2.0.3)

**v2.0.3 includes Rust backend optimizations** that further improve SyntheticDiD performance:

| Estimator | v2.0 (10K scale) | v2.0.3 (10K scale) | Speedup | vs R |
|-----------|------------------|-------------------|---------|------|
| BasicDiD/TWFE | 0.011s | **0.010s** | 1.1x | **4x faster than R** |
| CallawaySantAnna | 0.109s | **0.145s** | 0.8x | **5x faster than R** |
| SyntheticDiD (Pure) | 19.5s | **19.5s** | 1.0x | 57x faster than R |
| SyntheticDiD (Rust) | 2.6s | **2.6s** | 1.0x | **429x faster than R** |

**20K Scale Results** (new in v2.0.3 benchmarks):

| Estimator | Python Pure (s) | Python Rust (s) | R (s) | Rust vs R |
|-----------|-----------------|-----------------|-------|-----------|
| BasicDiD/TWFE | 0.022 | 0.025 | 0.050 | **2x** |
| CallawaySantAnna | 0.366 | 0.373 | 1.559 | **4x** |
| SyntheticDiD | 137.3 | **10.9** | 2451.0 | **225x** |

### What Changed in v2.0.3

1. **Cholesky factorization** for symmetric positive-definite matrix inversion (~2x faster for well-conditioned matrices)
2. **Reduced bootstrap allocations** - Direct Array2 allocation eliminates Vec<Vec<f64>> intermediate
3. **Vectorized variance computation** - HC1 meat uses BLAS-accelerated matrix operations
4. **Webb lookup table** - Faster Webb distribution weight generation
5. **Rayon chunk size tuning** - Reduced parallel scheduling overhead

---

## Results Achieved (v1.4.0)

**Phase 1 is complete.** Pure Python optimizations exceeded all targets:

| Estimator | v1.3 (10K scale) | v1.4 (10K scale) | Speedup | vs R |
|-----------|------------------|------------------|---------|------|
| BasicDiD/TWFE | 0.835s | **0.011s** | **76x** | **4.2x faster than R** |
| CallawaySantAnna | 2.234s | **0.109s** | **20x** | **7.2x faster than R** |
| SyntheticDiD | 32.6s | N/A | N/A | 37x faster than R |

### What Was Implemented

1. **Unified `linalg.py` backend** (`diff_diff/linalg.py`)
   - `solve_ols()` - scipy lstsq with gelsy LAPACK driver
   - `compute_robust_vcov()` - Vectorized cluster-robust SE via pandas groupby
   - Single optimization point for all estimators

2. **CallawaySantAnna optimizations** (`staggered.py`)
   - `_precompute_structures()` - Pre-computed wide-format outcome matrix, cohort masks
   - `_compute_att_gt_fast()` - Vectorized ATT(g,t) using numpy (23x faster)
   - `_generate_bootstrap_weights_batch()` - Batch weight generation
   - Vectorized bootstrap using matrix operations (26x faster)

3. **TWFE optimization** (`twfe.py`)
   - Cached groupby indexes for within-transformation

4. **All estimators migrated** to unified backend
   - `estimators.py`, `twfe.py`, `staggered.py`, `triple_diff.py`, `synthetic_did.py`, `sun_abraham.py`, `utils.py`

---

## Original Problem Statement

Benchmark comparisons showed that while diff-diff was competitive or faster than R for small datasets, performance degraded significantly at scale:

| Scale | BasicDiD Python | R (fixest) | Ratio |
|-------|-----------------|------------|-------|
| Small (<1K obs) | 0.003s | 0.041s | Python 16x faster |
| 5K (40-200K obs) | 0.180s | 0.046s | R 4x faster |
| 10K (100-500K obs) | 0.835s | 0.049s | R 17x faster |

| Scale | CallawaySantAnna Python | R (did) | Ratio |
|-------|-------------------------|---------|-------|
| Small | 0.048s | 0.077s | Python 1.6x faster |
| 5K | 0.793s | 0.382s | R 2x faster |
| 10K | 2.234s | 0.816s | R 2.7x faster |

Note: SyntheticDiD is already 37-1600x faster than R's synthdid package.

## Root Cause Analysis

### 1. OLS Solver (`estimators.py`)

Current implementation uses `np.linalg.lstsq` with default settings:
- General-purpose LAPACK driver (gelsd) rather than faster alternatives
- Preceded by expensive `matrix_rank()` check (O(min(n,k)^3))
- NumPy may not link to optimized BLAS

### 2. Cluster-Robust Standard Errors (`utils.py`)

Loop-based implementation:
```python
for cluster in unique_clusters:
    mask = cluster_ids == cluster  # O(n) per cluster
    ...
```
- O(n * n_clusters) complexity
- Creates boolean mask array for each cluster
- No vectorization or parallelization

### 3. Within-Transformation (`twfe.py`)

Multiple groupby operations:
```python
for var in variables:
    unit_means = data.groupby(unit)[var].transform("mean")
    time_means = data.groupby(time)[var].transform("mean")
    ...
```
- Multiple passes over data per variable
- No caching of groupby indexes
- Not using alternating projections algorithm

### 4. CallawaySantAnna Nested Loops (`staggered.py`)

```python
for g in treatment_groups:
    for t in valid_periods:
        att_gt = self._compute_att_gt(...)
```
- Repeated DataFrame indexing (`.set_index()`, `.loc[]`, `.isin()`) for each (g,t)
- No pre-computation of outcome changes
- Influence function dictionaries created per (g,t)

## Optimization Strategy

### Phase 1: Pure Python Optimizations (No New Dependencies)

Quick wins that improve performance without adding dependencies.

#### 1.1 Vectorized Cluster-Robust SE

Replace loop with vectorized groupby:
```python
scores = X * residuals[:, np.newaxis]
cluster_scores = pd.DataFrame(scores).groupby(cluster_ids).sum()
meat = cluster_scores.values.T @ cluster_scores.values
```

**Expected speedup:** 5-10x for SE computation

#### 1.2 scipy.linalg.lstsq with Optimized Driver

```python
from scipy.linalg import lstsq
coefficients = lstsq(X, y, lapack_driver='gelsy',
                     overwrite_a=True, overwrite_b=True,
                     check_finite=False)[0]
```

**Expected speedup:** 1.2-1.5x for OLS

#### 1.3 Cache Groupby Indexes

Create groupby objects once and reuse:
```python
unit_grouper = data.groupby(unit, sort=False)
time_grouper = data.groupby(time, sort=False)
```

**Expected speedup:** 1.5-2x for demeaning

#### 1.4 Pre-compute CallawaySantAnna Data Structures

Pivot to wide format once, pre-compute all period changes:
```python
outcome_wide = data.pivot(index=unit, columns=time, values=outcome)
changes = {(t0, t1): outcome_wide[t1] - outcome_wide[t0] for ...}
```

**Expected speedup:** 3-5x for CallawaySantAnna

### Phase 2: Compiled Backend

Implement performance-critical components in a compiled language for maximum speed.

#### Backend Options: Rust vs C++

We have two viable options for a compiled backend. Both can achieve near-identical performance; the choice depends on team expertise and maintenance considerations.

##### Option A: Rust with PyO3

**Pros:**
- **Memory safety by design** - No segfaults, buffer overflows, or data races; compiler catches these at build time
- **Modern tooling** - Cargo package manager + maturin makes wheel building straightforward
- **Zero-copy NumPy interop** - rust-numpy crate provides direct array access without copying
- **Easy parallelism** - rayon crate makes parallel iteration trivial (`.par_iter()`)
- **Growing ecosystem** - Used by polars, pyfixest, cryptography, orjson, ruff
- **Low per-call overhead** - Research shows PyO3 has ~0.14ms overhead vs NumPy's ~3.5ms for simple operations
- **Single toolchain** - `cargo build` works the same on all platforms

**Cons:**
- **Learning curve** - Rust's ownership model takes time to learn
- **Smaller scientific ecosystem** - Fewer numerical libraries than C++ (though ndarray and faer are mature)
- **Slower compilation** - Rust compiles slower than C++
- **Newer language** - Less institutional knowledge, fewer Stack Overflow answers

**Key dependencies:** `pyo3`, `rust-numpy`, `ndarray`, `faer` (linear algebra), `rayon` (parallelism)

##### Option B: C++ with pybind11

**Pros:**
- **Mature ecosystem** - Eigen, Armadillo, Intel MKL, OpenBLAS all native C++
- **Familiar to more developers** - Larger pool of contributors
- **Proven in scientific Python** - NumPy, SciPy, scikit-learn, pandas all use C/C++ extensions
- **Excellent Eigen integration** - pybind11 has built-in support for Eigen matrices
- **Faster compilation** - C++ compiles faster than Rust
- **More optimization resources** - Decades of C++ performance tuning knowledge

**Cons:**
- **Memory safety risks** - Segfaults, buffer overflows, use-after-free possible; harder to debug
- **Manual memory management** - Must carefully manage lifetimes, especially with Python GC interaction
- **Complex build systems** - CMake configuration, compiler flags, platform-specific issues
- **Copy overhead by default** - pybind11 copies arrays unless carefully configured with `py::array_t`
- **Manual GIL management** - Easy to deadlock or corrupt state if GIL not handled correctly
- **Platform differences** - MSVC vs GCC vs Clang have different behaviors and flags

**Key dependencies:** `pybind11`, `Eigen` (linear algebra), `OpenMP` or `TBB` (parallelism)

##### Comparison Summary

| Factor | Rust (PyO3) | C++ (pybind11) |
|--------|-------------|----------------|
| Memory safety | Compile-time guarantees | Runtime risks |
| Build tooling | Cargo + maturin (simple) | CMake + scikit-build (complex) |
| NumPy interop | Zero-copy via rust-numpy | Zero-copy possible but tricky |
| Parallelism | rayon (trivial) | OpenMP/TBB (more boilerplate) |
| Linear algebra | faer, ndarray-linalg | Eigen, MKL, OpenBLAS |
| Ecosystem maturity | Growing | Established |
| Learning curve | Steeper (ownership) | Moderate (but footguns) |
| Wheel building | maturin-action (simple) | cibuildwheel (more config) |
| Debug experience | Good (cargo, clippy) | Variable (platform-dependent) |

##### Recommendation

**Rust with PyO3** is the recommended approach because:

1. **pyfixest validates this for our exact domain** - They use Rust/PyO3 for fixed effects econometrics
2. **Memory safety prevents production bugs** - No risk of segfaults in user code
3. **maturin simplifies distribution** - Single command builds wheels for all platforms
4. **rayon makes parallelization trivial** - Critical for bootstrap and cluster SE

However, **C++ is a viable alternative** if:
- Team has stronger C++ expertise
- Need to integrate with existing C++ econometrics code
- Want to leverage Eigen's mature linear algebra

#### Graceful Degradation

```python
try:
    from diff_diff._rust_backend import solve_ols_clustered
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

def _fit_ols(self, X, y, cluster_ids=None):
    if _HAS_RUST and self.backend == 'rust':
        return solve_ols_clustered(X, y, cluster_ids)
    else:
        # Existing NumPy implementation
        ...
```

#### Components to Implement in Rust

| Component | Current Bottleneck | Rust Benefit |
|-----------|-------------------|--------------|
| Cluster-robust SE | O(n * clusters) loop | rayon parallel iteration |
| Within-transformation | Multiple groupby passes | Single-pass with hash tables |
| OLS solving | NumPy lstsq overhead | faer or direct LAPACK |
| Bootstrap resampling | Sequential iterations | Embarrassingly parallel |
| ATT(g,t) computation | Repeated DataFrame indexing | Pre-indexed sparse structures |

#### Architecture by Backend

##### Rust Layout

```
diff_diff/
├── estimators.py          # Python API (unchanged)
├── _rust_backend/         # Compiled Rust module
│   └── ...
└── _fallback.py           # Pure Python fallback

src/                       # Rust source (Cargo workspace)
├── Cargo.toml
├── lib.rs
├── ols.rs                 # OLS with cluster SE
├── demeaning.rs           # Alternating projections
├── bootstrap.rs           # Parallel bootstrap
└── staggered.rs           # ATT(g,t) computation

pyproject.toml             # maturin build config
```

##### C++ Layout

```
diff_diff/
├── estimators.py          # Python API (unchanged)
├── _cpp_backend/          # Compiled C++ module
│   └── ...
└── _fallback.py           # Pure Python fallback

cpp/                       # C++ source
├── CMakeLists.txt
├── src/
│   ├── module.cpp         # pybind11 bindings
│   ├── ols.cpp            # OLS with cluster SE
│   ├── ols.hpp
│   ├── demeaning.cpp      # Within transformation
│   ├── demeaning.hpp
│   ├── bootstrap.cpp      # Parallel bootstrap
│   └── bootstrap.hpp
└── extern/
    └── eigen/             # Eigen submodule (or system install)

pyproject.toml             # scikit-build-core config
```

#### Distribution

##### Rust (maturin)

```yaml
# .github/workflows/wheels.yml
- uses: PyO3/maturin-action@v1
  with:
    command: build
    args: --release --out dist
```

- Simple single-action CI configuration
- Use abi3 stable ABI for Python version-independent wheels
- Cross-compilation via `--target` flag

##### C++ (cibuildwheel)

```yaml
# .github/workflows/wheels.yml
- uses: pypa/cibuildwheel@v2
  env:
    CIBW_BUILD: "cp39-* cp310-* cp311-* cp312-*"
```

- More configuration required for CMake integration
- Need to handle OpenMP linking per-platform
- Consider vcpkg or conan for dependency management

Both approaches build wheels for:
- Linux (manylinux2014, x86_64 and aarch64)
- macOS (x86_64 and ARM64)
- Windows (x86_64)

## Implementation Roadmap

| Phase | Scope | Effort | Expected Speedup |
|-------|-------|--------|------------------|
| 1.1 | Vectorize cluster SE | 1-2 days | 5-10x (SE only) |
| 1.2 | scipy lstsq optimization | 1 day | 1.2-1.5x (OLS) |
| 1.3 | Cache groupby indexes | 1 day | 1.5-2x (demeaning) |
| 1.4 | Pre-compute CS structures | 2-3 days | 3-5x (CS) |
| 2.1 | Rust cluster SE | 1-2 weeks | 10-50x (SE) |
| 2.2 | Rust parallel bootstrap | 1 week | 5-20x (bootstrap) |
| 2.3 | Rust demeaning | 2 weeks | 3-10x (TWFE) |
| 2.4 | Rust OLS solver | 2 weeks | Match R |
| 2.5 | Rust staggered ATT | 2-3 weeks | 5-10x (CS) |
| 2.6 | CI/CD wheel building | 1 week | N/A |

## Outcomes

### Phase 1 Results (v1.4.0) ✅

**Exceeded all targets:**

- BasicDiD @ 10K: 0.835s → **0.011s** (76x improvement, 4.2x faster than R)
- CallawaySantAnna @ 10K: 2.2s → **0.109s** (20x improvement, 7.2x faster than R)
- Bootstrap inference: 26x faster via vectorization

### Phase 2 (Rust Backend) - Optional Future Work

No longer required for R parity. May be pursued for:
- Further optimization at extreme scales (100K+ units)
- Parallel bootstrap across CPU cores
- Memory efficiency for very large datasets

## References

### Rust Backend

- [PyO3 User Guide](https://pyo3.rs/) - Rust bindings for Python
- [rust-numpy](https://github.com/PyO3/rust-numpy) - Zero-copy NumPy interop
- [maturin](https://github.com/PyO3/maturin) - Build and publish Rust Python packages
- [faer](https://github.com/sarah-ek/faer-rs) - Pure Rust linear algebra (competitive with MKL)
- [Polars](https://github.com/pola-rs/polars) - Example of Rust/Python hybrid architecture
- [pyfixest](https://github.com/py-econometrics/pyfixest) - Rust backend for fixed effects econometrics

### C++ Backend

- [pybind11 documentation](https://pybind11.readthedocs.io/) - C++ bindings for Python
- [pybind11 Eigen integration](https://pybind11.readthedocs.io/en/stable/advanced/cast/eigen.html) - Zero-copy with Eigen
- [Eigen](https://eigen.tuxfamily.org/) - C++ linear algebra library
- [scikit-build-core](https://scikit-build-core.readthedocs.io/) - CMake integration for Python packages
- [cibuildwheel](https://cibuildwheel.readthedocs.io/) - Build wheels for all platforms

### General

- [fixest demeaning algorithm](https://rdrr.io/cran/fixest/man/demeaning_algo.html) - Reference implementation
- [PyO3 vs C performance comparison](https://www.alphaxiv.org/overview/2507.00264v1) - Academic benchmark
- [Making Python 100x faster with Rust](https://ohadravid.github.io/posts/2023-03-rusty-python/) - Practical tutorial
