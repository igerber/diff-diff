Benchmarks: Validation Against R Packages
=========================================

This document presents validation benchmarks comparing diff-diff against
established R packages for difference-in-differences analysis.

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
5. **Multiple Scales**: Test at small (200-400 obs), 1K, 5K, and 10K unit scales
6. **Replicated Timing**: 10 replications per benchmark to report mean ± std
7. **Reproducible Seed**: Benchmarks use seed 20260111 for data generation

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
     - diff-diff
     - R fixest
     - Difference
   * - ATT
     - 5.112
     - 5.112
     - < 1e-10
   * - SE
     - 0.183
     - 0.183
     - 0.0%
   * - Time (s)
     - 0.002
     - 0.035
     - **17.9x faster**

**Validation**: PASS - Results are numerically identical.

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
     - diff-diff
     - R did
     - Difference
   * - ATT
     - 2.519
     - 2.519
     - < 1e-10
   * - SE
     - 0.062
     - 0.063
     - 2.3%
   * - Time (s)
     - 0.005
     - 0.070
     - **14.0x faster**

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

We benchmarked performance across multiple dataset scales with 10 replications
each to provide mean ± std timing statistics.

.. note::

   **v1.4.0 Performance Improvements**: diff-diff v1.4.0 introduced major
   performance optimizations including a unified linear algebra backend
   (``diff_diff/linalg.py``) with scipy's optimized gelsy LAPACK driver,
   vectorized cluster-robust standard errors, and optimized CallawaySantAnna
   bootstrap using matrix operations. These improvements make diff-diff
   **faster than R at all scales**.

Summary by Scale
~~~~~~~~~~~~~~~~

**Small Scale** (400-1,600 observations):

.. list-table::
   :header-rows: 1
   :widths: 30 25 25 20

   * - Estimator
     - Python (s)
     - R (s)
     - Speedup
   * - BasicDiD/TWFE
     - 0.002 ± 0.000
     - 0.035 ± 0.001
     - **17.9x**
   * - CallawaySantAnna
     - 0.005 ± 0.000
     - 0.070 ± 0.001
     - **14.0x**

**1K Scale** (6,000-10,000 observations):

.. list-table::
   :header-rows: 1
   :widths: 30 25 25 20

   * - Estimator
     - Python (s)
     - R (s)
     - Speedup
   * - BasicDiD/TWFE
     - 0.003 ± 0.001
     - 0.035 ± 0.001
     - **12.5x**
   * - CallawaySantAnna
     - 0.012 ± 0.000
     - 0.113 ± 0.002
     - **9.6x**

**5K Scale** (40,000-60,000 observations):

.. list-table::
   :header-rows: 1
   :widths: 30 25 25 20

   * - Estimator
     - Python (s)
     - R (s)
     - Speedup
   * - BasicDiD/TWFE
     - 0.006 ± 0.003
     - 0.038 ± 0.002
     - **6.1x**
   * - CallawaySantAnna
     - 0.055 ± 0.001
     - 0.339 ± 0.002
     - **6.2x**

**10K Scale** (100,000-150,000 observations):

.. list-table::
   :header-rows: 1
   :widths: 30 25 25 20

   * - Estimator
     - Python (s)
     - R (s)
     - Speedup
   * - BasicDiD/TWFE
     - 0.010 ± 0.000
     - 0.041 ± 0.001
     - **4.1x**
   * - CallawaySantAnna
     - 0.155 ± 0.002
     - 0.730 ± 0.004
     - **4.7x**

Key Observations
~~~~~~~~~~~~~~~~

1. **diff-diff is faster than R at all scales**: Following v1.4.0 optimizations,
   diff-diff now outperforms R packages across all dataset sizes for BasicDiD/TWFE
   and CallawaySantAnna estimators.

2. **BasicDiD/TWFE**: diff-diff is 4-18x faster than R's ``fixest::feols``.
   The speedup is greatest at small scales (17.9x) and remains substantial
   at large scales (4.1x at 10K observations).

3. **CallawaySantAnna**: diff-diff is 5-14x faster than R's ``did::att_gt``
   using analytical SEs. At small scales (14x speedup), pure Python overhead
   is minimal; at larger scales the gap narrows but remains substantial (4.7x).

4. **Scaling behavior**: Both estimators show sub-linear scaling in diff-diff.
   At 10K scale (150K observations for CallawaySantAnna), estimation completes
   in ~150ms with analytical SEs.

Performance Optimization Details
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The v1.4.0 performance improvements came from:

1. **Unified ``linalg.py`` backend**: Single optimized OLS/SE implementation
   using scipy's gelsy LAPACK driver (QR-based, faster than SVD)

2. **Vectorized cluster-robust SE**: Eliminated O(n × clusters) loop with
   pandas groupby aggregation

3. **Pre-computed data structures** (CallawaySantAnna): Wide-format outcome
   matrix and cohort masks computed once, reused across all ATT(g,t) calculations

4. **Vectorized bootstrap** (CallawaySantAnna): Matrix operations instead of
   nested loops, batch weight generation

Why is diff-diff Fast?
~~~~~~~~~~~~~~~~~~~~~~

1. **Optimized LAPACK**: scipy's gelsy driver for least squares
2. **Vectorized operations**: NumPy/pandas for matrix operations and aggregations
3. **Efficient memory access**: Pre-computed structures avoid repeated data reshaping
4. **Pure Python overhead minimized**: Hot paths use compiled NumPy/scipy routines

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

   # Run all benchmarks at all scales with 10 replications
   python benchmarks/run_benchmarks.py --all --scale all --replications 10

   # Run specific estimator at specific scale
   python benchmarks/run_benchmarks.py --estimator callaway --scale 1k --replications 10
   python benchmarks/run_benchmarks.py --estimator synthdid --scale small --replications 5
   python benchmarks/run_benchmarks.py --estimator basic --scale 5k --replications 10

   # Available scales: small, 1k, 5k, 10k, all
   # Default: small (backward compatible)

   # Generate synthetic data only (use seed for reproducibility)
   python benchmarks/run_benchmarks.py --generate-data-only --scale all --seed 20260111

The benchmarks in this documentation were run with seed 20260111 (date-based:
2026-01-11) for reproducibility.

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
