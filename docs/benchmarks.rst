Benchmarks: Validation Against R Packages
=========================================

This document presents validation benchmarks comparing diff-diff against
established R packages for difference-in-differences analysis. As of v2.0.0,
diff-diff includes an optional Rust backend for accelerated computation.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------

diff-diff is validated against the following R packages:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - diff-diff Estimator
     - R Package
     - Reference
   * - ``DifferenceInDifferences``
     - ``fixest::feols``
     - Standard OLS with interaction
   * - ``CallawaySantAnna``
     - ``did::att_gt``
     - Callaway & Sant'Anna (2021)
   * - ``SyntheticDiD``
     - ``synthdid::synthdid_estimate``
     - Arkhangelsky et al. (2021)

Methodology
-----------

Validation Approach
~~~~~~~~~~~~~~~~~~~

1. **Synthetic Data**: Generate data with known true effects using
   ``generate_did_data()`` from diff_diff.prep
2. **Identical Inputs**: Both Python and R estimators receive the same CSV data
3. **JSON Interchange**: R scripts output JSON for comparison
4. **Automated Comparison**: Python script validates numerical equivalence
5. **Multiple Scales**: Test at small (200-400 obs), 1K, 5K, 10K, and 20K unit scales
6. **Replicated Timing**: 3 replications per benchmark to report mean ± std
7. **Reproducible Seed**: Benchmarks use seed 42 for data generation
8. **Three-Way Comparison**: Compare R, Python (pure NumPy/SciPy), and Python (Rust backend)

Tolerance Thresholds
~~~~~~~~~~~~~~~~~~~~

- **Point estimates (ATT)**: Absolute difference < 1e-4 or relative < 1%
- **Standard errors**: Relative difference < 10%
- **Confidence intervals**: Must overlap

Benchmark Results
-----------------

Summary Table
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 20 20 15 20

   * - Estimator
     - ATT Diff
     - SE Rel Diff
     - CI Overlap
     - Status
   * - BasicDiD/TWFE
     - < 1e-10
     - 0.0%
     - Yes
     - **PASS**
   * - CallawaySantAnna
     - < 1e-10
     - < 1%
     - Yes
     - **PASS**
   * - SyntheticDiD
     - 0.011
     - 3.1%
     - Yes
     - **PASS**

Basic DiD Results
~~~~~~~~~~~~~~~~~

**Data**: 100 units, 4 periods, true ATT = 5.0 (small scale)

.. list-table::
   :header-rows: 1

   * - Metric
     - diff-diff (Pure)
     - diff-diff (Rust)
     - R fixest
     - Difference
   * - ATT
     - 5.112
     - 5.112
     - 5.112
     - < 1e-10
   * - SE
     - 0.183
     - 0.183
     - 0.183
     - 0.0%
   * - Time (s)
     - 0.002
     - 0.002
     - 0.041
     - **22x faster**

**Validation**: PASS - Results are numerically identical across all implementations.

Synthetic DiD Results
~~~~~~~~~~~~~~~~~~~~~

**Data**: 50 units (40 control, 10 treated), 20 periods, true ATT = 4.0

.. list-table::
   :header-rows: 1

   * - Metric
     - diff-diff
     - R synthdid
     - Difference
   * - ATT
     - 3.851
     - 3.840
     - 0.011 (0.3%)
   * - SE
     - 0.106
     - 0.103
     - 3.1%
   * - Time (s)
     - 0.017
     - 7.49
     - **433x faster**

**Validation**: PASS - Both ATT and SE estimates match closely. Both implementations
use placebo-based variance estimation (R's Algorithm 4 from Arkhangelsky et al. 2021).

The small SE difference (3.1%) is due to different unit/time weight optimization
algorithms:

- diff-diff uses projected gradient descent
- R synthdid uses Frank-Wolfe optimization with adaptive regularization

This leads to slightly different weights, which propagate to the placebo estimates.

Callaway-Sant'Anna Results
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Data**: 200 units, 8 periods, 3 treatment cohorts, dynamic effects (small scale)

.. list-table::
   :header-rows: 1

   * - Metric
     - diff-diff (Pure)
     - diff-diff (Rust)
     - R did
     - Difference
   * - ATT
     - 2.519
     - 2.519
     - 2.519
     - < 1e-10
   * - SE
     - 0.062
     - 0.062
     - 0.063
     - 2.3%
   * - Time (s)
     - 0.005
     - 0.005
     - 0.071
     - **14x faster**

**Validation**: PASS - Both point estimates and standard errors match R closely.

**Key findings from investigation:**

1. **Individual ATT(g,t) effects match perfectly** (~1e-11 difference)
2. **Never-treated coding**: R's ``did`` package requires ``first_treat=Inf``
   for never-treated units. diff-diff accepts ``first_treat=0``. The benchmark
   converts 0 to Inf for R compatibility.
3. **Standard errors**: As of v1.5.0, analytical SEs use influence function
   aggregation (matching R's approach), resulting in < 3% SE difference across
   all scales. Both analytical and bootstrap inference now match R closely.

Performance Comparison
----------------------

We benchmarked performance across multiple dataset scales with 3 replications
each to provide mean ± std timing statistics. As of v2.0.0, we compare three
implementations:

- **R**: Reference implementation (fixest, did packages)
- **Python (Pure)**: diff-diff with NumPy/SciPy only (no Rust backend)
- **Python (Rust)**: diff-diff with optional Rust backend enabled

.. note::

   **v2.0.0 Rust Backend**: diff-diff v2.0.0 introduces an optional Rust backend
   for accelerated computation. The Rust backend provides significant speedups
   for **SyntheticDiD** (4-8x faster than pure Python), which uses custom Rust
   implementations for synthetic weight computation and simplex projection.
   For **BasicDiD** and **CallawaySantAnna**, the Rust backend provides minimal
   additional speedup since these estimators primarily use OLS and variance
   computations that are already highly optimized in NumPy/SciPy via BLAS/LAPACK.

Three-Way Performance Summary
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**BasicDiD/TWFE Results:**

.. list-table::
   :header-rows: 1
   :widths: 12 15 18 18 12 12

   * - Scale
     - R (s)
     - Python Pure (s)
     - Python Rust (s)
     - Rust/R
     - Rust/Pure
   * - small
     - 0.035
     - 0.002
     - 0.002
     - **18x**
     - 1.1x
   * - 1k
     - 0.037
     - 0.003
     - 0.003
     - **14x**
     - 1.1x
   * - 5k
     - 0.038
     - 0.008
     - 0.006
     - **7x**
     - 1.4x
   * - 10k
     - 0.041
     - 0.010
     - 0.011
     - **4x**
     - 0.9x
   * - 20k
     - 0.050
     - 0.026
     - 0.025
     - **2x**
     - 1.1x

**CallawaySantAnna Results:**

.. list-table::
   :header-rows: 1
   :widths: 12 15 18 18 12 12

   * - Scale
     - R (s)
     - Python Pure (s)
     - Python Rust (s)
     - Rust/R
     - Rust/Pure
   * - small
     - 0.071
     - 0.005
     - 0.005
     - **14.1x**
     - 1.0x
   * - 1k
     - 0.114
     - 0.012
     - 0.012
     - **9.4x**
     - 1.0x
   * - 5k
     - 0.341
     - 0.055
     - 0.056
     - **6.1x**
     - 1.0x
   * - 10k
     - 0.726
     - 0.156
     - 0.155
     - **4.7x**
     - 1.0x
   * - 20k
     - 1.464
     - 0.404
     - 0.411
     - **3.6x**
     - 1.0x

**SyntheticDiD Results:**

.. list-table::
   :header-rows: 1
   :widths: 12 15 18 18 12 12

   * - Scale
     - R (s)
     - Python Pure (s)
     - Python Rust (s)
     - Rust/R
     - Rust/Pure
   * - small
     - 8.18
     - 0.015
     - 0.004
     - **2234x**
     - **4.0x**
   * - 1k
     - 110.4
     - 0.068
     - 0.100
     - **1104x**
     - 0.7x
   * - 5k
     - 511.1
     - 3.017
     - 0.688
     - **743x**
     - **4.4x**
   * - 10k
     - 1462.7
     - 19.56
     - 2.59
     - **565x**
     - **7.6x**

.. note::

   **SyntheticDiD Performance**: diff-diff achieves **565x to 2234x speedup** over
   R's synthdid package. At 10k scale, R takes ~24 minutes while Python Rust
   completes in 2.6 seconds. The Rust backend provides **4-8x additional speedup**
   over pure Python for SyntheticDiD due to optimized simplex projection and
   synthetic weight computation. ATT estimates differ slightly due to different
   weight optimization algorithms (projected gradient descent vs Frank-Wolfe),
   but confidence intervals overlap.

Dataset Sizes
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 12 22 22 22 22

   * - Scale
     - BasicDiD
     - CallawaySantAnna
     - SyntheticDiD
     - Observations
   * - small
     - 100 × 4
     - 200 × 8
     - 50 × 20
     - 400 - 1,600
   * - 1k
     - 1,000 × 6
     - 1,000 × 10
     - 1,000 × 30
     - 6,000 - 30,000
   * - 5k
     - 5,000 × 8
     - 5,000 × 12
     - 5,000 × 40
     - 40,000 - 200,000
   * - 10k
     - 10,000 × 10
     - 10,000 × 15
     - 10,000 × 50
     - 100,000 - 500,000
   * - 20k
     - 20,000 × 12
     - 20,000 × 18
     - N/A
     - 240,000 - 360,000

Key Observations
~~~~~~~~~~~~~~~~

1. **diff-diff is dramatically faster than R**:

   - **BasicDiD/TWFE**: 2-18x faster than R
   - **CallawaySantAnna**: 4-14x faster than R
   - **SyntheticDiD**: 565-2234x faster than R (R takes 24 minutes at 10k scale!)

2. **Rust backend benefit depends on the estimator**:

   - **SyntheticDiD**: Rust provides **4-8x speedup** over pure Python due to
     optimized simplex projection and synthetic weight computation
   - **BasicDiD/CallawaySantAnna**: Rust provides minimal benefit (~1x) since
     these estimators use OLS/variance computations already optimized in NumPy/SciPy

3. **When to use Rust backend**:

   - **SyntheticDiD**: Recommended - provides significant speedup (4-8x)
   - **Bootstrap inference**: May help with parallelized iterations
   - **BasicDiD/CallawaySantAnna**: Optional - pure Python is equally fast

4. **Scaling behavior**: Both Python implementations show excellent scaling.
   At 10K scale (500K observations for SyntheticDiD), Rust completes in
   ~2.6 seconds vs ~20 seconds for pure Python vs ~24 minutes for R.

5. **No Rust required for most use cases**: Users without Rust/maturin can
   install diff-diff and get full functionality with excellent performance.
   For BasicDiD and CallawaySantAnna, pure Python achieves the same speed as Rust.
   Only SyntheticDiD benefits significantly from the Rust backend.

Performance Optimization Details
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The performance improvements come from:

1. **Unified ``linalg.py`` backend**: Single optimized OLS/SE implementation
   using scipy's gelsy LAPACK driver (QR-based, faster than SVD)

2. **Vectorized cluster-robust SE**: Eliminated O(n × clusters) loop with
   pandas groupby aggregation

3. **Pre-computed data structures** (CallawaySantAnna): Wide-format outcome
   matrix and cohort masks computed once, reused across all ATT(g,t) calculations

4. **Vectorized bootstrap** (CallawaySantAnna): Matrix operations instead of
   nested loops, batch weight generation

5. **Optional Rust backend** (v2.0.0): PyO3-based Rust extension for compute-intensive
   operations (OLS, robust variance, bootstrap weights, simplex projection)

Why is diff-diff Fast?
~~~~~~~~~~~~~~~~~~~~~~

1. **Optimized LAPACK**: scipy's gelsy driver for least squares
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
     - 0.0117
     - 0.0118
     - **< 1%**
   * - Time (10 reps)
     - 0.003s ± 0.000s
     - 0.039s ± 0.006s
     - **14.4x faster**

**Key Findings:**

1. **Point estimates match exactly**: The overall ATT of -0.039951 is identical
   between diff-diff and R's ``did`` package, validating the core estimation logic.

2. **Standard errors match**: As of v1.5.0, analytical SEs use influence function
   aggregation (matching R's approach), resulting in < 1% difference. Both point
   estimates and standard errors now match R's ``did`` package.

3. **Performance**: diff-diff is ~14x faster than R on this real-world dataset,
   consistent with the synthetic data benchmarks at small scale.

This validation on real-world data with known published results confirms that
diff-diff produces correct estimates that match the reference R implementation.

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

Running Benchmarks
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Run all benchmarks at small scale
   python benchmarks/run_benchmarks.py --all

   # Run all benchmarks at all scales with 3 replications
   python benchmarks/run_benchmarks.py --all --scale all --replications 3

   # Run specific estimator at specific scale
   python benchmarks/run_benchmarks.py --estimator callaway --scale 1k --replications 3
   python benchmarks/run_benchmarks.py --estimator synthdid --scale small --replications 3
   python benchmarks/run_benchmarks.py --estimator basic --scale 20k --replications 3

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

- **SyntheticDiD**: Both point estimates (0.3% diff) and standard errors (3.1% diff)
  match R closely. Use ``variance_method="placebo"`` (default) to match R's
  inference. Results are fully validated.

- **CallawaySantAnna**: Group-time effects (ATT(g,t)) are reliable. Overall
  ATT aggregation may differ from R due to weighting choices. When comparing
  to R ``did`` package, verify aggregation settings match.

Known Differences
~~~~~~~~~~~~~~~~~

1. **Inference Methods**: diff-diff defaults to analytical SEs; R ``did``
   defaults to multiplier bootstrap. Enable bootstrap in diff-diff for
   direct comparison.

2. **Aggregation Weights**: Overall ATT is a weighted average of ATT(g,t).
   Weighting schemes may differ between implementations.

3. **Weight Optimization**: SyntheticDiD uses different optimization algorithms
   for unit/time weights (projected gradient descent vs Frank-Wolfe). This leads
   to slightly different weights but equivalent ATT estimates.

References
----------

.. [CS2021] Callaway, B., & Sant'Anna, P. H. (2021). Difference-in-Differences
   with multiple time periods. *Journal of Econometrics*, 225(2), 200-230.

.. [AHIW2021] Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W.,
   & Wager, S. (2021). Synthetic Difference-in-Differences. *American Economic
   Review*, 111(12), 4088-4118.

.. [RR2023] Rambachan, A., & Roth, J. (2023). A More Credible Approach to
   Parallel Trends. *Review of Economic Studies*, 90(5), 2555-2591.
