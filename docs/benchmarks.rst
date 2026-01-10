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

Summary
~~~~~~~

All diff-diff estimators are significantly faster than their R equivalents:

.. list-table::
   :header-rows: 1
   :widths: 30 20 20 30

   * - Estimator
     - Python (s)
     - R (s)
     - Speedup
   * - BasicDiD/TWFE
     - 0.002
     - 0.039
     - **17x**
   * - CallawaySantAnna
     - 0.044
     - 0.068
     - **1.6x**
   * - SyntheticDiD
     - 0.017
     - 7.49
     - **433x** ± 12x

Key Observations
~~~~~~~~~~~~~~~~

1. **BasicDiD/TWFE**: diff-diff is 17x faster for simple DiD estimation.
   Both packages produce numerically identical results.

2. **CallawaySantAnna**: diff-diff is 1.6x faster with identical ATT estimates.
   Individual ATT(g,t) effects match perfectly. SE matches within 5.6% when
   using multiplier bootstrap (default analytical SEs use different formulas).

3. **SyntheticDiD**: diff-diff is **433x faster** (measured over 5 runs:
   mean 433x, std 12x) while producing equivalent estimates. R spends ~7.5s
   on placebo variance estimation alone, while diff-diff completes the entire
   estimation in 0.017s. SE now matches within 3.1% since both use placebo
   variance estimation.

Why is diff-diff Faster?
~~~~~~~~~~~~~~~~~~~~~~~~

1. **Pure NumPy/SciPy**: No R overhead or bridge communication
2. **Optimized algorithms**: Vectorized operations throughout
3. **Minimal dependencies**: No heavy statistical frameworks
4. **Efficient memory**: Direct array operations without copying
5. **Efficient placebo variance**: For SyntheticDiD, the placebo SE computation
   permutes indices and renormalizes weights using vectorized operations, rather
   than re-running the full optimization for each replication. This accounts for
   the 433x speedup - R's placebo method takes ~7.5s while diff-diff takes ~0.017s.

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

   # Run all benchmarks
   python benchmarks/run_benchmarks.py --all

   # Run specific estimator
   python benchmarks/run_benchmarks.py --estimator callaway
   python benchmarks/run_benchmarks.py --estimator synthdid
   python benchmarks/run_benchmarks.py --estimator basic

   # Generate synthetic data only
   python benchmarks/run_benchmarks.py --generate-data-only

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
