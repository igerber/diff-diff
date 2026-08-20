.. meta::
   :description: Validation benchmarks comparing diff-diff against R packages (did, synthdid, fixest) and Stata (teffects, lpdid, did_imputation, jwdid/csdid, reghdfe, lwdid). Coefficient accuracy, standard error comparison, and performance metrics.
   :keywords: difference-in-differences benchmark, DiD validation R, DiD validation Stata, python econometrics accuracy, did package comparison

Benchmarks
==========

This document presents validation benchmarks comparing diff-diff against
established R packages for difference-in-differences analysis. Released
wheels bundle a Rust backend that is used automatically; the "Pure" arm in
the tables below runs the same wheel with the backend disabled
(``DIFF_DIFF_BACKEND=python``).

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------

diff-diff is validated primarily against established R packages, plus Stata
where no runnable R reference exists:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - diff-diff Estimator
     - Reference Package
     - Reference
   * - ``DifferenceInDifferences``
     - ``fixest::feols``
     - Standard OLS with interaction
   * - ``TwoWayFixedEffects``
     - ``fixest::feols`` (absorbed FE)
     - Within-transformed unit + period fixed effects
   * - ``CallawaySantAnna``
     - ``did::att_gt``
     - Callaway & Sant'Anna (2021)
   * - ``MultiPeriodDiD``
     - ``fixest::feols``
     - Event study with treatment × period interactions
   * - ``SyntheticDiD``
     - ``synthdid::synthdid_estimate``
     - Arkhangelsky et al. (2021)
   * - ``LPDiD`` (regression-adjustment SE)
     - Stata ``teffects ra ... atet``
     - Dube, Girardi, Jorda & Taylor (2025); no runnable R analogue
   * - ``LPDiD`` (non-absorbing SEs)
     - Stata ``lpdid, nonabsorbing(...)`` (the authors' package, end-to-end)
     - Dube, Girardi, Jorda & Taylor (2025) §4.2; first external anchor for the
       non-absorbing reweighted SE and pooled windows. Eq. 12 on all surfaces;
       Eq. 13 at post horizons + pooled post on a convention-neutral subsample
       (three package convention differences measured and divergence-gated — see
       the methodology registry, LPDiD Deviation 4)
   * - ``ImputationDiD`` (leave-one-out SE)
     - Stata ``did_imputation, leaveout``
     - Borusyak, Jaravel & Spiess (2024) Supp. App. A.9; no runnable R analogue.
       Covers all three ``aux_partition`` values via ``avgeffectsby(Ei t | Ei | K)``,
       plus an unbalanced subsample where the cohort partition genuinely diverges
   * - ``ETWFE`` / ``CallawaySantAnna`` (cross-check)
     - Stata ``jwdid`` / ``csdid``
     - Wooldridge (2021) ETWFE and CS via the authors' Stata implementations
   * - Clustered CR1 ``K_reference`` (shared linalg)
     - Stata ``reghdfe``
     - Disconnected-panel absorbed-FE rank convention
   * - ``LWDiD`` (validation arm precommitted; estimator in review, PR #588)
     - Stata ``lwdid`` (authors' package)
     - Lee & Wooldridge (2025, 2026); small-N exact + RI + event-study bootstrap

Methodology
-----------

Validation Approach
~~~~~~~~~~~~~~~~~~~

1. **Synthetic Data**: Generate data with known true effects using
   ``generate_did_data()`` from diff_diff.prep
2. **Identical Inputs**: Both Python and the reference implementation (R, or Stata
   where no runnable R analogue exists) receive the same data
3. **JSON Interchange**: reference generators (R / Stata) output JSON for comparison
4. **Automated Comparison**: Python script validates numerical equivalence
5. **Multiple Scales**: Test at small (200-400 obs), 1K, 5K, 10K, and 20K unit scales
6. **Replicated Timing**: multiple fresh-subprocess replications per benchmark
   (8 for fast cells, 4 for the slow SyntheticDiD scales), each with an untimed
   in-process warm-up fit; the first replication is excluded and the **median**
   of the remaining replications is published (full distributions are retained
   in the committed results artifact)
7. **Reproducible Seed**: Benchmarks use seed 42 for data generation
8. **Three-Way Comparison**: Compare R, Python (pure NumPy/SciPy), and Python (Rust backend)

Tolerance Thresholds
~~~~~~~~~~~~~~~~~~~~

- **Point estimates (ATT)**: Absolute difference < 1e-4 or relative < 1%
- **Standard errors**: Relative difference < 10%. Exception: SyntheticDiD
  placebo SEs use a documented 35% Monte Carlo-bounded gate - both
  implementations estimate the placebo variance by simulation and R's
  placebo permutation is unseeded, so SEs agree in distribution rather
  than draw-by-draw (see the SyntheticDiD methodology registry note;
  the deterministic Frank-Wolfe ATT is gated at 1e-8)
- **Confidence intervals**: Must overlap

Benchmark Results
-----------------

Summary Table
~~~~~~~~~~~~~

.. refresh-table-start: summary

.. list-table::
   :header-rows: 1
   :widths: 25 20 20 15 20

   * - Estimator
     - ATT Diff
     - SE Rel Diff
     - CI Overlap
     - Status
   * - BasicDiD
     - < 5e-11
     - 0.0%
     - Yes
     - **PASS**
   * - TWFE (absorbed FE)
     - < 6e-12
     - 0.1%
     - Yes
     - **PASS**
   * - MultiPeriodDiD
     - < 2e-12
     - 0.0%
     - Yes
     - **PASS**
   * - CallawaySantAnna
     - < 4e-11
     - 0.0%
     - Yes
     - **PASS**
   * - SyntheticDiD
     - < 4e-11
     - 11.5%
     - Yes
     - **PASS**

SyntheticDiD SE differences reflect Monte Carlo dispersion of the
placebo variance (R's placebo permutation is unseeded, so the two
implementations agree in distribution, not draw-by-draw): its SE is
gated at a 35% relative bound with R's rep-to-rep SE values recorded
in the committed results artifact, while the deterministic
Frank-Wolfe ATT is gated at 1e-8. All other estimators use analytical
SEs gated at the tolerances above. See the SyntheticDiD methodology
registry note (benchmark SE gate is Monte Carlo-bounded).

.. refresh-table-end: summary

Basic DiD Results
~~~~~~~~~~~~~~~~~

**Data**: 100 units, 4 periods, true ATT = 5.0 (small scale)

.. refresh-table-start: accuracy_basic

.. list-table::
   :header-rows: 1
   :widths: 16 21 21 21 21

   * - Metric
     - diff-diff (Pure)
     - diff-diff (Rust)
     - R fixest
     - Difference
   * - ATT
     - 5.112
     - 5.112
     - 5.112
     - < 4e-11
   * - SE
     - 0.183
     - 0.183
     - 0.183
     - 0.0%
   * - Time (s)
     - 0.0005
     - 0.0005
     - 0.0040
     - **7.6x faster** (rust)

.. refresh-table-end: accuracy_basic

**Validation**: PASS - Results are numerically identical across all implementations.

MultiPeriodDiD Results
~~~~~~~~~~~~~~~~~~~~~~

**Data**: 200 units, 8 periods (4 pre, 4 post), true ATT = 3.0 (small scale)

.. refresh-table-start: accuracy_multiperiod

.. list-table::
   :header-rows: 1
   :widths: 16 21 21 21 21

   * - Metric
     - diff-diff (Pure)
     - diff-diff (Rust)
     - R fixest
     - Difference
   * - ATT
     - 2.912
     - 2.912
     - 2.912
     - < 2e-12
   * - SE
     - 0.158
     - 0.158
     - 0.158
     - 0.0%
   * - Time (s)
     - 0.0040
     - 0.0040
     - 0.0056
     - **1.4x faster** (pure)

.. refresh-table-end: accuracy_multiperiod

**Validation**: PASS - Both average ATT and all period-level effects match R's
``fixest::feols(outcome ~ treated * time_f | unit)`` to machine precision. The
regression includes unit fixed effects (absorbed via ``| unit`` in R, within-
transformation via ``absorb=["unit"]`` in Python) and treatment × period
interactions with cluster-robust SEs.

Synthetic DiD Results
~~~~~~~~~~~~~~~~~~~~~

**Data**: 50 units (40 control, 10 treated), 20 periods, true ATT = 4.0

.. refresh-table-start: accuracy_synthdid

.. list-table::
   :header-rows: 1
   :widths: 16 21 21 21 21

   * - Metric
     - diff-diff (Pure)
     - diff-diff (Rust)
     - R synthdid
     - Difference
   * - ATT
     - 3.840
     - 3.840
     - 3.840
     - < 2e-11
   * - SE
     - 0.113
     - 0.113
     - 0.101
     - 11.5%
   * - Time (s)
     - 15.4
     - 0.139
     - 7.65
     - **55.0x faster** (rust)

.. refresh-table-end: accuracy_synthdid

**Validation**: PASS - ATT estimates are numerically identical across all
implementations. Both diff-diff and R's synthdid use Frank-Wolfe optimization
with two-pass sparsification and auto-computed regularization (``zeta_omega``,
``zeta_lambda``), producing identical unit and time weights (reproduced at
< 1e-8 by the benchmark harness's id-aligned per-unit comparison; see the
SyntheticDiD methodology registry note). Both use placebo-based variance
estimation (Algorithm 4 from Arkhangelsky et al. 2021).

The SE difference (11.5% at small scale, ~3% at 1k/5k in this capture) is
Monte Carlo dispersion of the placebo procedure, which randomly permutes
control units to construct pseudo-treated groups: R's placebo permutation is
unseeded, so the two implementations agree in distribution rather than
draw-by-draw. R's rep-to-rep SE values are recorded in the committed results
artifact, and the benchmark gates this at a documented 35% bound (see the
SyntheticDiD methodology registry note).

Callaway-Sant'Anna Results
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Data**: 200 units, 8 periods, 3 treatment cohorts, dynamic effects (small scale)

.. refresh-table-start: accuracy_callaway

.. list-table::
   :header-rows: 1
   :widths: 16 21 21 21 21

   * - Metric
     - diff-diff (Pure)
     - diff-diff (Rust)
     - R did
     - Difference
   * - ATT
     - 2.519
     - 2.519
     - 2.519
     - < 3e-11
   * - SE
     - 0.063
     - 0.063
     - 0.063
     - 0.0%
   * - Time (s)
     - 0.0028
     - 0.0028
     - 0.013
     - **4.6x faster** (pure)

.. refresh-table-end: accuracy_callaway

**Validation**: PASS - Both point estimates and standard errors match R exactly.

**Key findings from investigation:**

1. **Individual ATT(g,t) effects match perfectly** (~1e-11 difference)
2. **Never-treated coding**: R's ``did`` package requires ``first_treat=Inf``
   for never-treated units. diff-diff accepts ``first_treat=0``. The benchmark
   converts 0 to Inf for R compatibility.
3. **Standard errors**: As of v2.0.2, analytical SEs match R's ``did`` package
   exactly (0.0% difference). The weight influence function (wif) formula was
   corrected to match R's implementation, achieving numerical equivalence across
   all dataset scales.

**Event study per-event-time validation:**

As of v2.1.0, event study SEs include the WIF adjustment matching R's
``did::aggte(..., type="dynamic")``. Validation targets:

- Per-event-time point estimates: match R's ``aggte(..., type="dynamic")`` to <1e-10
- Per-event-time analytical SEs (``bstrap=FALSE``): match R with WIF included
- Per-event-time bootstrap SEs (``bstrap=TRUE``): consistent with analytical
- Simultaneous confidence bands (``cband=TRUE``): sup-t critical value matches R

Performance Comparison
----------------------

We benchmarked performance across multiple dataset scales. Reported timings
are medians of repeated fresh-subprocess replications with an untimed warm-up
fit on BOTH sides (R's byte-compiler JIT is kept out of the timing window; see
the environment block below for the full protocol). We compare three
implementations:

- **R**: Reference implementation (fixest, did packages)
- **Python (Pure)**: diff-diff with NumPy/SciPy only (no Rust backend)
- **Python (Rust)**: diff-diff with optional Rust backend enabled

.. refresh-table-start: environment

.. rubric:: Benchmark environment (2026-07 refresh)

- **Captured**: 2026-07-10T22:59:43+00:00
- **Hardware**: Apple M4 Max, 36 GB RAM, macOS 26.5.2 (arm64)
- **diff-diff**: 3.7.0 released wheel from PyPI (Rust backend + Apple Accelerate), Python 3.14.4, NumPy 2.5.1, pandas 3.0.3
- **R**: R version 4.5.2 (2025-10-31); did 2.5.1, fixest 0.14.2, synthdid 0.0.9 (installed at capture)
- **Threads**: No arm is thread-restricted: R runs at fixest/data.table defaults; diff-diff wheels run at Accelerate/rayon defaults. Thread-count env vars (RAYON/OMP/OPENBLAS/VECLIB/MKL/data.table) are stripped from every benchmark subprocess and R runs under --vanilla (no user/site .Rprofile/.Renviron), so package defaults are enforced, not assumed. Per-arm thread counts are recorded in each result's metadata.
- **Protocol**: Each replication is a fresh subprocess run strictly sequentially (one benchmark process on the machine at a time) with an untimed in-process warm-up fit before the timed fit. The first replication is additionally excluded from statistics. Published statistic: median of the counted replications. Arms with CV > 10% are rerun once and flagged if still noisy.

.. refresh-table-end: environment

.. note::

   **Rust Backend**: released wheels bundle the Rust backend (used
   automatically). It is transformative for **SyntheticDiD** - 110x faster
   than pure Python at small scale and 4-12x at 1k-5k in this capture, making
   Rust the fastest implementation at every scale - via Gram-accelerated /
   allocation-free Frank-Wolfe solvers and a batched placebo variance path.
   For **BasicDiD**, **TWFE**, and **CallawaySantAnna**, Rust and pure Python
   are near parity: those estimators are dominated by OLS/variance kernels
   already optimized in NumPy/SciPy via BLAS/LAPACK.

   Pre-built wheels on macOS and Linux link platform-optimized BLAS libraries
   (Apple Accelerate and OpenBLAS respectively) across all Rust-accelerated
   code paths. Windows wheels use pure Rust with no external dependencies.

Three-Way Performance Summary
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. refresh-table-start: perf_basic

**BasicDiD (interaction OLS, clustered):**

.. list-table::
   :header-rows: 1
   :widths: 12 15 18 18 12 12

   * - Scale
     - R (s)
     - Python Pure (s)
     - Python Rust (s)
     - Pure/R
     - Rust/R
   * - small
     - 0.0040
     - 0.0005
     - 0.0005
     - **7.5x**
     - **7.6x**
   * - 1k
     - 0.0042
     - 0.0011
     - 0.0010
     - **4.0x**
     - **4.2x**
   * - 5k
     - 0.0056
     - 0.0039
     - 0.0038
     - 1.4x
     - 1.5x
   * - 10k
     - 0.0077
     - 0.0086
     - 0.0086
     - 0.9x
     - 0.9x
   * - 20k
     - 0.013
     - 0.020
     - 0.019
     - 0.7x
     - 0.7x

**TWFE (absorbed unit + post fixed effects, clustered):**

.. list-table::
   :header-rows: 1
   :widths: 12 15 18 18 12 12

   * - Scale
     - R (s)
     - Python Pure (s)
     - Python Rust (s)
     - Pure/R
     - Rust/R
   * - small
     - 0.0046
     - 0.0011
     - 0.0012
     - **4.0x**
     - **3.9x**
   * - 1k
     - 0.0049
     - 0.0020
     - 0.0019
     - **2.5x**
     - **2.6x**
   * - 5k
     - 0.0072
     - 0.0066
     - 0.0058
     - 1.1x
     - 1.2x
   * - 10k
     - 0.011
     - 0.014
     - 0.012
     - 0.8x
     - 0.9x
   * - 20k
     - 0.020
     - 0.032
     - 0.027
     - 0.6x
     - 0.7x

.. refresh-table-end: perf_basic

**CallawaySantAnna Results:**

.. refresh-table-start: perf_callaway

.. list-table::
   :header-rows: 1
   :widths: 12 15 18 18 12 12

   * - Scale
     - R (s)
     - Python Pure (s)
     - Python Rust (s)
     - Pure/R
     - Rust/R
   * - small
     - 0.013
     - 0.0028
     - 0.0028
     - **4.6x**
     - **4.6x**
   * - 1k
     - 0.026
     - 0.0043
     - 0.0043
     - **6.0x**
     - **6.0x**
   * - 5k
     - 0.078
     - 0.0098
     - 0.0098
     - **8.0x**
     - **8.0x**
   * - 10k
     - 0.306
     - 0.020
     - 0.020
     - **15x**
     - **15x**
   * - 20k
     - 0.621
     - 0.042
     - 0.042
     - **15x**
     - **15x**

.. refresh-table-end: perf_callaway

**SyntheticDiD Results:**

.. refresh-table-start: perf_synthdid

.. list-table::
   :header-rows: 1
   :widths: 12 15 18 18 12 12

   * - Scale
     - R (s)
     - Python Pure (s)
     - Python Rust (s)
     - Pure/R
     - Rust/R
   * - small
     - 7.65
     - 15.4
     - 0.139
     - 0.5x
     - **55x**
   * - 1k
     - 101
     - 65.0
     - 5.59
     - 1.6x
     - **18x**
   * - 5k
     - 467
     - 106
     - 25.2
     - **4.4x**
     - **19x**

.. refresh-table-end: perf_synthdid

.. note::

   **SyntheticDiD Performance**: the Rust backend (the default in released
   wheels) is **18-55x faster than R's synthdid** at matched 200-replication
   placebo variance - at 5k scale R takes ~7.8 minutes while Rust completes
   in 25 seconds. Pure Python ranges from 0.5x (small scale - slower than R)
   to 4.4x (5k) at the same matched placebo-replication counts; earlier versions of this
   page compared a 50-replication Python arm against R's 200 and overstated
   the pure-Python advantage. ATT estimates are numerically identical
   (< 1e-10) and unit/time weights reproduce R at < 1e-8 (id-aligned
   comparison), since both implementations use the same Frank-Wolfe optimizer
   with two-pass sparsification. The Rust backend uses a Gram-accelerated
   Frank-Wolfe solver for time weights (per-iteration cost O(T0) instead of
   O(N×T0)) and an allocation-free solver for unit weights.

Dataset Sizes
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 10 18 18 18 18 18

   * - Scale
     - BasicDiD
     - MultiPeriodDiD
     - CallawaySantAnna
     - SyntheticDiD
     - Observations
   * - small
     - 100 × 4
     - 200 × 8
     - 200 × 8
     - 50 × 20
     - 400 - 1,600
   * - 1k
     - 1,000 × 6
     - 1,000 × 10
     - 1,000 × 10
     - 1,000 × 30
     - 6,000 - 30,000
   * - 5k
     - 5,000 × 8
     - 5,000 × 12
     - 5,000 × 12
     - 5,000 × 40
     - 40,000 - 200,000
   * - 10k
     - 10,000 × 10
     - 10,000 × 12
     - 10,000 × 15
     - 10,000 × 50
     - 100,000 - 500,000
   * - 20k
     - 20,000 × 12
     - 20,000 × 16
     - 20,000 × 18
     - 20,000 × 60
     - 240,000 - 1,200,000

TWFE (absorbed FE) benchmarks reuse the BasicDiD datasets at every scale.

Key Observations
~~~~~~~~~~~~~~~~

1. **Where diff-diff wins, the advantage grows with scale**:

   - **CallawaySantAnna**: 4.6x faster than R at small scale rising to
     **15.5x at 10k units (150k observations) and 14.8x at 20k units
     (360k observations)** - the 2026 scaling work (O(n_units) aggregation influence functions, fused
     bootstrap, per-cell solver fast paths) pays off exactly where compute
     matters, with exact SE parity (0.0% difference) maintained.
   - **SyntheticDiD (Rust backend, the default install)**: **18-55x faster
     than R** at matched placebo-replication counts.

2. **The honest small-regression story**: BasicDiD and TWFE interaction/
   absorbed OLS cells all complete in under 35 milliseconds on both sides.
   diff-diff is 2.5-7.6x faster up to 5k; at 10k-20k, warmed-up fixest is
   1.3-1.5x faster than us. At these absolute times the difference is
   immaterial in practice - fixest is exceptionally well optimized for
   simple regressions, and our performance work targets the estimators
   where runtimes are measured in seconds or minutes.

3. **Rust backend benefit depends on the estimator**: transformative for
   SyntheticDiD (fastest implementation at every scale; 110x vs pure Python
   at small scale) and near parity with pure Python for BasicDiD/TWFE/
   CallawaySantAnna, whose hot paths are already BLAS-bound in NumPy/SciPy.
   Released wheels bundle Rust, so no toolchain is needed.

4. **SyntheticDiD pure Python vs R at equal placebo-replication counts**: 0.5x (small,
   slower than R) to 4.4x (5k). Earlier versions of this page compared
   unequal placebo-replication counts and overstated the pure-Python
   advantage; install the default wheel and the Rust backend makes the
   question moot.

5. **Every published number is gated**: ATT/SE tolerances per rendered arm,
   CI overlap, per-period / per-(g,t) / event-study / group effect surfaces,
   SyntheticDiD id-aligned weight identity, real-data known answers, and
   replication determinism all hard-gate publication (a failed cell cannot
   render). See the environment block above for the capture context.

6. **CallawaySantAnna accuracy and speed**: CallawaySantAnna
   achieves both exact numerical accuracy (0.0% SE difference from R) AND
   superior performance (4.6-15.5x faster than R) through vectorized weight
   influence function (WIF) computation using NumPy matrix operations.

Performance Optimization Details
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The performance improvements come from:

1. **Unified ``linalg.py`` backend**: Single optimized OLS/SE implementation
   using an equilibrated SVD solve (gelsd-parity) with certified stage-0
   rank detection

2. **Vectorized cluster-robust SE**: Eliminated O(n × clusters) loop with
   pandas groupby aggregation

3. **Pre-computed data structures** (CallawaySantAnna): Wide-format outcome
   matrix and cohort masks computed once, reused across all ATT(g,t) calculations

4. **Vectorized bootstrap** (CallawaySantAnna): Matrix operations instead of
   nested loops, batch weight generation

5. **Vectorized WIF computation** (CallawaySantAnna, v2.0.3): Weight influence
   function computation uses NumPy matrix operations instead of O(n_units × n_keepers)
   nested loops. The indicator matrix, if1/if2 matrices, and wif contribution are
   computed using broadcasting and matrix multiplication: ``wif_contrib = wif_matrix @ effects``

6. **Optional Rust backend** (v2.0.0): PyO3-based Rust extension for compute-intensive
   operations (OLS, robust variance, bootstrap weights, simplex projection)

Why is diff-diff Fast?
~~~~~~~~~~~~~~~~~~~~~~

1. **Optimized LAPACK**: equilibrated SVD least squares with certified
   rank detection
2. **Vectorized operations**: NumPy/pandas for matrix operations and aggregations
3. **Efficient memory access**: Pre-computed structures avoid repeated data reshaping
4. **Pure Python overhead minimized**: Hot paths use compiled NumPy/scipy routines
5. **Optional Rust acceleration**: Native code for bootstrap and optimization algorithms

Real-World Data Validation
--------------------------

In addition to synthetic data benchmarks, we validate diff-diff against the
**MPDTA (Minimum Wage and Teen Employment)** dataset - the canonical benchmark
used in Callaway & Sant'Anna (2021) and the R ``did`` package.

MPDTA Dataset
~~~~~~~~~~~~~

The MPDTA dataset contains county-level teen employment data with staggered
minimum wage policy changes:

- **500 counties** across 5 years (2003-2007)
- **2,500 observations** total
- **4 treatment cohorts**: Never-treated (309), 2004 (20), 2006 (40), 2007 (131)
- **Outcome**: Log teen employment (``lemp``)
- **Source**: Built into R's ``did`` package

Results Comparison
~~~~~~~~~~~~~~~~~~

.. refresh-table-start: mpdta

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Metric
     - diff-diff
     - R did
     - Difference
   * - ATT
     - -0.039951
     - -0.039951
     - **0** (exact match)
   * - SE (analytical)
     - 0.0120
     - 0.0120
     - **< 0.1%**
   * - Time (median of 7)
     - 0.0028s
     - 0.013s
     - **4.6x faster**

.. refresh-table-end: mpdta

**Key Findings:**

1. **Point estimates match exactly**: The overall ATT of -0.039951 is identical
   between diff-diff and R's ``did`` package, validating the core estimation logic.

2. **Standard errors match exactly**: As of v2.0.2, analytical SEs use the corrected
   weight influence function formula, achieving 0.0% difference from R's ``did``
   package. Both point estimates and standard errors are numerically equivalent.

3. **Performance**: diff-diff is ~5x faster than R on this real-world dataset
   (both complete in milliseconds). Performance advantages grow with scale
   (see the CallawaySantAnna performance table above).

This validation on real-world data with known published results confirms that
diff-diff produces correct estimates that match the reference R implementation.

Survey Real-Data Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

In addition to synthetic-data survey cross-validation (see
``test_survey_r_crossvalidation.py``), diff-diff's survey variance is validated
against R's ``survey`` package using three real federal survey datasets. All
comparisons match to machine precision (differences < 1e-10).

**Datasets:**

.. list-table::
   :header-rows: 1
   :widths: 15 20 15 20 30

   * - Dataset
     - Source
     - Size
     - Survey Design
     - Policy Context
   * - API (apistrat)
     - R ``survey`` package
     - 200 schools
     - Strata + FPC + weights
     - California school accountability (PSAA 1999)
   * - NHANES
     - CDC/NCHS
     - 2,946 adults
     - Strata + PSU + weights (nest)
     - ACA young adult coverage provision (2010)
   * - RECS 2020
     - U.S. EIA
     - 2,000 households
     - 60 JK1 replicate weights
     - Residential energy consumption survey

**Suite A — API Dataset (TSL Variance):**

.. list-table::
   :header-rows: 1
   :widths: 10 30 20 20 20

   * - Test
     - Design Variant
     - ATT Gap
     - SE Gap
     - df
   * - A1
     - Strata + FPC + weights
     - 2.1e-12
     - 3.2e-11 (0.0000%)
     - Exact
   * - A2
     - Strata + weights (no FPC)
     - 2.1e-12
     - 5.3e-11 (0.0000%)
     - Exact
   * - A3
     - Weights only
     - 2.1e-12
     - 2.7e-11 (0.0000%)
     - Exact
   * - A4
     - TWFE (strata + FPC + weights)
     - 2.1e-12 (ATT only)
     - n/a (TWFE absorbs unit FE)
     - Exact
   * - A5
     - Subpopulation (elementary)
     - 1.5e-11
     - 7.5e-12 (0.0000%)
     - Differs (see note)
   * - A6
     - Covariates (meals, ell)
     - 2.2e-12
     - 8.4e-12 (0.0000%)
     - Exact
   * - A7
     - Fay's BRR replicates (rho=0.3)
     - 2.1e-12
     - 7.7e-11 (0.0000%)
     - Exact

**Suite B — NHANES (TSL with Strata + PSU + nest=TRUE):**

.. list-table::
   :header-rows: 1
   :widths: 10 30 20 20 20

   * - Test
     - Design Variant
     - ATT Gap
     - SE Gap
     - df
   * - B1
     - Strata + PSU + weights
     - 4.6e-13
     - 2.3e-14 (0.0000%)
     - Exact (31)
   * - B2
     - Covariates (gender, poverty)
     - 4.9e-13
     - 2.3e-13 (0.0000%)
     - Exact
   * - B3
     - Weights only
     - 4.6e-13
     - 2.6e-13 (0.0000%)
     - Exact
   * - B4
     - Subpopulation (female)
     - 1.1e-13
     - 8.7e-14 (0.0000%)
     - Exact

**Suite C — RECS 2020 (JK1 Replicate Weights):**

.. list-table::
   :header-rows: 1
   :widths: 10 30 20 20 20

   * - Test
     - Model
     - Coef Gap
     - SE Gap
     - df
   * - C1
     - TOTALBTU ~ KOWNRENT
     - 1.5e-11
     - 2.0e-11 (0.0000%)
     - Exact (59)
   * - C2
     - + TYPEHUQ + REGIONC
     - 3.8e-10
     - 2.9e-11 (0.0000%)
     - Exact (59)

**Key Findings:**

1. **Machine-precision agreement** on ATT, SE, df, and CI wherever directly
   comparable — differences are < 1e-10 (floating-point rounding only).
   Tolerances are set to 1e-8 in the test suite.

2. **All survey design features validated with real data:** stratification, PSU
   clustering, FPC corrections, probability weight normalization, nested PSU
   handling (``nest=TRUE``), subpopulation analysis, covariate adjustment,
   Fay's BRR (212 replicates), and JK1 replicate weight variance.

3. **Known differences:** A4 (TWFE) validates ATT only — SE differs because
   TWFE absorbs unit fixed effects. A5 (subpopulation) validates ATT/SE but
   df differs: ``subpopulation()`` preserves all strata (df=397) while R's
   ``subset()`` drops empty strata (df=199). This is a documented deviation
   (see REGISTRY.md); the diff-diff approach is conservative per Lumley (2004).

Survey Estimator Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Four additional estimators are validated against R's ``survey::svyglm()`` using
synthetic staggered-adoption and DDD datasets. Each estimator reduces to a WLS
regression under survey weights, so the R comparison fits the equivalent
``svyglm()`` model and compares coefficients and standard errors.

**Data:** 150-unit staggered panel (5 periods, 4 strata, 10 PSUs, FPC) with
cohorts at *t* = 3 and *t* = 4; 200-observation DDD cross-section (4 strata,
10 PSUs, FPC). Both generated with seed 42.

.. list-table::
   :header-rows: 1
   :widths: 8 18 30 15 15 14

   * - Test
     - Estimator
     - R Comparison
     - Coef Gap
     - SE Gap
     - Tolerance
   * - S1
     - ``ImputationDiD``
     - ``svyglm()`` on control-only (Omega_0) FE regression; covariate coefficients
     - < 1e-10
     - 0.00%
     - 1.5%
   * - S2
     - ``StackedDiD``
     - ``svyglm()`` on stacked dataset with Q-weight x survey weight composition
     - < 1e-10
     - 0.77%
     - 1.5%
   * - S3
     - ``SunAbraham``
     - ``svyglm()`` with cohort x period interactions; IW-aggregated ATT
     - < 1e-11
     - 0.00%
     - 1.5%
   * - S4
     - ``TripleDifference``
     - ``svyglm()`` three-way interaction (``group:partition:time``)
     - < 1e-10
     - 0.36%
     - 1.5%

**Key details:**

- **S1** validates the WLS building block that ``ImputationDiD`` uses internally
  (control-only regression with absorbed unit + time FE and time-varying
  covariates). A companion smoke test confirms ``ImputationDiD.fit()`` produces
  finite ATT/SE under survey weights.
- **S2** replicates the full stacking pipeline in R: sub-experiment construction,
  sample-share Q-weight computation, Q x survey weight composition with
  normalization, then ``svyglm()`` on the stacked data with strata/PSU structure.
  The 0.77% SE gap arises because R omits FPC on the stacked data while Python
  re-resolves the full survey design.
- **S3** compares both individual cohort x relative-time effects and the
  IW-aggregated overall ATT (with survey-weighted cohort masses and delta-method
  SE via the vcov submatrix).
- **S4** exploits the algebraic equivalence between the pairwise DDD
  decomposition (``estimation_method="reg"``, no covariates) and the three-way
  interaction coefficient from a single OLS regression.

Reproducing Survey Estimator Validation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Generate golden values
   Rscript benchmarks/R/benchmark_survey_estimators.R

   # Run validation tests
   pytest tests/test_survey_estimator_validation.py -v

Reproducing Survey Real-Data Validation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # 1. Generate API golden values (no download needed — data ships with R)
   Rscript benchmarks/R/benchmark_realdata_api.R

   # 2. Download and process NHANES data from CDC
   python benchmarks/scripts/download_nhanes.py
   Rscript benchmarks/R/benchmark_realdata_nhanes.R

   # 3. Download and subset RECS 2020 from EIA
   python benchmarks/scripts/download_recs.py
   Rscript benchmarks/R/benchmark_realdata_recs.R

   # 4. Run validation tests
   pytest tests/test_survey_real_data.py -v

Reproducing Benchmarks
----------------------

Prerequisites
~~~~~~~~~~~~~

1. Install R (>= 4.0):

   .. code-block:: bash

      # macOS
      brew install r

2. Install R packages:

   .. code-block:: bash

      Rscript benchmarks/R/requirements.R

3. Install diff-diff:

   .. code-block:: bash

      pip install -e ".[dev]"

4. (Optional) Stata, only to regenerate the committed Stata goldens (``LPDiD``
   regression-adjustment SE, ``LPDiD`` non-absorbing SEs, ``ImputationDiD``
   leave-one-out SE, the ETWFE/CS cross-check, the ``reghdfe`` K_reference
   convention, and the LWDiD authors'-package parity). The goldens are committed,
   so this is not needed to run the test suite. The ``LPDiD`` RA arm uses the
   **native** ``teffects`` command; the other arms depend on SSC packages
   (``did_imputation``/``reghdfe``/``ftools``/``require``,
   ``drdid``/``csdid``/``jwdid``/``hdfe``, ``lwdid``,
   ``lpdid``/``boottest``/``egenmore``/``listreg``) — install them once via
   ``benchmarks/stata/requirements.do`` (the generators do not auto-install):

   .. code-block:: bash

      # macOS, StataSE 19 (binary not on PATH by default)
      STATA=/Applications/Stata/StataSE.app/Contents/MacOS/stata-se
      $STATA -b do benchmarks/stata/requirements.do            # one-time SSC install
      $STATA -b do benchmarks/stata/generate_lpdid_ra_golden.do
      $STATA -b do benchmarks/stata/generate_lpdid_nonabsorbing_golden.do
      $STATA -b do benchmarks/stata/generate_imputation_loo_golden.do
      $STATA -b do benchmarks/stata/generate_etwfe_cs_golden.do
      $STATA -b do benchmarks/stata/generate_reghdfe_kref_golden.do
      $STATA -b do benchmarks/stata/generate_lwdid_golden.do   # warm-up step: see benchmarks/stata/README.md

Running Benchmarks
~~~~~~~~~~~~~~~~~~

The published tables on this page are produced by the gated refresh harness
(fresh venv with the released wheel, warm-ups, medians, fail-closed
publication gates):

.. code-block:: bash

   # One-time: create the pinned-wheel venv and preflight it
   python benchmarks/refresh_2026_07/run_refresh.py --setup

   # Full gated run (idle machine recommended; SDID R cells dominate)
   python benchmarks/refresh_2026_07/run_refresh.py

   # Regenerate the marker-bounded tables on this page from the results
   python benchmarks/refresh_2026_07/gen_benchmark_tables.py

The legacy harness below remains available for quick accuracy comparisons
(it is NOT the source of the published tables):

.. code-block:: bash

   # Run all benchmarks at small scale
   python benchmarks/run_benchmarks.py --all

   # Run all benchmarks at all scales with 3 replications
   python benchmarks/run_benchmarks.py --all --scale all --replications 3

   # Run specific estimator at specific scale
   python benchmarks/run_benchmarks.py --estimator callaway --scale 1k --replications 3
   python benchmarks/run_benchmarks.py --estimator synthdid --scale small --replications 3
   python benchmarks/run_benchmarks.py --estimator basic --scale 20k --replications 3
   python benchmarks/run_benchmarks.py --estimator multiperiod --scale small --replications 3

   # Available scales: small, 1k, 5k, 10k, 20k, all
   # Default: small (backward compatible)

   # Generate synthetic data only
   python benchmarks/run_benchmarks.py --generate-data-only --scale all

The benchmarks run both pure Python and Rust backends automatically, producing
a three-way comparison table (R vs Python Pure vs Python Rust).

Output
~~~~~~

Results are saved to:

- ``benchmarks/results/accuracy/`` - JSON files with estimates
- ``benchmarks/results/comparison_report.txt`` - Summary report

Interpretation Notes
--------------------

When to Trust Results
~~~~~~~~~~~~~~~~~~~~~

- **BasicDiD/TWFE**: Results are identical to R. Use with confidence.

- **MultiPeriodDiD**: Results are identical to R's ``fixest::feols`` with
  ``treated * time_f | unit`` interaction syntax (unit FE absorbed). Both average
  ATT and all period-level effects match to machine precision. Use with confidence.

- **SyntheticDiD**: Point estimates are numerically identical (< 1e-10 diff) and
  standard errors agree within placebo Monte Carlo dispersion. Both
  implementations use Frank-Wolfe optimization with identical unit and time
  weights (verified by id-aligned comparison in the benchmark harness). Use
  ``variance_method="placebo"`` (default) to match R's inference. Results are
  fully validated.

- **CallawaySantAnna**: Both group-time effects (ATT(g,t)) and overall ATT
  aggregation match R exactly. Standard errors are numerically equivalent
  (0.0% difference) as of v2.0.2.

Known Differences
~~~~~~~~~~~~~~~~~

1. **Inference Methods**: diff-diff defaults to analytical SEs; R ``did``
   defaults to multiplier bootstrap. Enable bootstrap in diff-diff for
   direct comparison.

2. **Aggregation Weights**: Overall ATT is a weighted average of ATT(g,t).
   Weighting schemes may differ between implementations.

3. **Placebo Variance**: SyntheticDiD SE estimates differ across
   implementations due to Monte Carlo variance in the placebo procedure
   (R's placebo permutation is unseeded). Point estimates and unit/time
   weights are numerically identical since both implementations use the
   same Frank-Wolfe optimizer (weights verified by id-aligned comparison
   in the benchmark harness).

