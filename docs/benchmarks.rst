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
     - 5.6%
     - Yes
     - **PASS**
   * - SyntheticDiD
     - 0.011
     - 3.1%
     - Yes
     - **PASS**

Basic DiD Results
~~~~~~~~~~~~~~~~~

**Data**: 100 units, 4 periods, true ATT = 5.0

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
     - 0.034
     - **14.5x faster**

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

**Data**: 200 units, 8 periods, 3 treatment cohorts, dynamic effects

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
     - 0.060
     - 0.063
     - 5.6%
   * - Time (s)
     - 0.044
     - 0.068
     - **1.6x faster**

**Validation**: PASS - Results match when using comparable inference methods.

**Key findings from investigation:**

1. **Individual ATT(g,t) effects match perfectly** (~1e-11 difference)
2. **Never-treated coding**: R's ``did`` package requires ``first_treat=Inf``
   for never-treated units. diff-diff accepts ``first_treat=0``. The benchmark
   converts 0 to Inf for R compatibility.
3. **Standard errors**: With multiplier bootstrap (``n_bootstrap=200``),
   Python SE (0.060) matches R SE (0.063) within 5.6%. The analytical SE
   formulas differ due to covariance handling in aggregation.

Performance Comparison
----------------------

We benchmarked performance across multiple dataset scales with 10 replications
each to provide mean ± std timing statistics.

Summary by Scale
~~~~~~~~~~~~~~~~

**Small Scale** (200-400 observations):

.. list-table::
   :header-rows: 1
   :widths: 30 25 25 20

   * - Estimator
     - Python (s)
     - R (s)
     - Speedup
   * - BasicDiD/TWFE
     - 0.003 ± 0.000
     - 0.041 ± 0.001
     - **16x**
   * - CallawaySantAnna
     - 0.048 ± 0.000
     - 0.077 ± 0.001
     - **1.6x**
   * - SyntheticDiD
     - 0.015 ± 0.000
     - 8.389 ± 0.589
     - **553x**

**1K Scale** (6,000-30,000 observations):

.. list-table::
   :header-rows: 1
   :widths: 30 25 25 20

   * - Estimator
     - Python (s)
     - R (s)
     - Speedup
   * - BasicDiD/TWFE
     - 0.011 ± 0.000
     - 0.043 ± 0.001
     - **4x**
   * - CallawaySantAnna
     - 0.162 ± 0.001
     - 0.129 ± 0.003
     - 0.8x (R faster)
   * - SyntheticDiD
     - 0.071 ± 0.001
     - 113.8 ± 2.6
     - **1613x**

**5K Scale** (40,000-200,000 observations):

.. list-table::
   :header-rows: 1
   :widths: 30 25 25 20

   * - Estimator
     - Python (s)
     - R (s)
     - Speedup
   * - BasicDiD/TWFE
     - 0.180 ± 0.001
     - 0.046 ± 0.001
     - 0.3x (R faster)
   * - CallawaySantAnna
     - 0.793 ± 0.008
     - 0.382 ± 0.004
     - 0.5x (R faster)
   * - SyntheticDiD
     - 3.130 ± 0.013
     - 556.1 ± 63.1
     - **178x**

**10K Scale** (100,000-500,000 observations):

.. list-table::
   :header-rows: 1
   :widths: 30 25 25 20

   * - Estimator
     - Python (s)
     - R (s)
     - Speedup
   * - BasicDiD/TWFE
     - 0.835 ± 0.003
     - 0.049 ± 0.001
     - 0.06x (R faster)
   * - CallawaySantAnna
     - 2.234 ± 0.011
     - 0.816 ± 0.006
     - 0.4x (R faster)
   * - SyntheticDiD
     - 32.6 ± 8.5
     - 1220.8 ± 30.7
     - **37x**

Key Observations
~~~~~~~~~~~~~~~~

1. **SyntheticDiD**: diff-diff maintains a substantial speed advantage at all scales,
   though the speedup ratio varies: 553x (small), 1613x (1K peak), 178x (5K), 37x (10K).
   The speedup peaks at 1K scale because Python's vectorized placebo variance estimation
   is most efficient at medium scales. At very large scales (10K), both implementations
   slow down, but Python remains 37x faster while R takes ~20 minutes per run.

2. **BasicDiD/TWFE**: At small scales (< 1K observations), diff-diff is 4-16x
   faster. However, at larger scales (5K+), R's ``fixest::feols`` with its
   highly optimized C++ backend becomes 3-17x faster.

3. **CallawaySantAnna**: At small scales, diff-diff is 1.6x faster. At 1K+
   observations, R's ``did::att_gt`` package scales better and becomes 1.3-2.7x
   faster. This reflects R's efficient compiled code for propensity score
   estimation and bootstrap inference.

Performance Scaling Pattern
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The benchmarks reveal an important pattern:

- **Small datasets (< 1K observations)**: Python is faster due to lower
  interpreter overhead and efficient NumPy operations.
- **Large datasets (5K+ observations)**: R packages with C++/Fortran backends
  (like ``fixest`` and ``did``) scale better for matrix operations.
- **SyntheticDiD**: Always dramatically faster in Python due to fundamentally
  different variance estimation algorithms.

This suggests diff-diff is optimal for:

- Exploratory analysis and prototyping
- Small to medium-sized datasets
- Synthetic DiD estimation at any scale
- Integration with Python ML/data science workflows

For production workloads with very large panel datasets (100K+ observations)
using BasicDiD or CallawaySantAnna, R may be more performant.

Why is diff-diff Faster for Small Datasets?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Pure NumPy/SciPy**: Lower overhead than R's interpreter
2. **Vectorized operations**: Efficient for small-medium matrices
3. **Minimal dependencies**: No heavy statistical frameworks loading
4. **Efficient placebo variance**: SyntheticDiD permutes indices and renormalizes
   weights using vectorized operations rather than re-running full optimization.

Why is R Faster for Large Datasets?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Compiled backends**: ``fixest`` and ``did`` use C++/Fortran for core operations
2. **Optimized linear algebra**: R links to highly tuned BLAS/LAPACK
3. **Memory-efficient algorithms**: Designed for large econometric datasets

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

   # Generate synthetic data only
   python benchmarks/run_benchmarks.py --generate-data-only --scale all

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
