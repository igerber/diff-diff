Comparison with R Packages
==========================

This guide compares diff-diff with popular R packages for DiD analysis, helping
users familiar with R transition to Python.

Overview
--------

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Feature
     - diff-diff (Python)
     - did (R)
     - Other R
   * - Basic DiD
     - ✅ ``DifferenceInDifferences``
     - ✅ ``att_gt``
     - ✅ ``fixest::feols``
   * - Staggered DiD
     - ✅ ``CallawaySantAnna``
     - ✅ ``att_gt``
     - ``did2s``, ``DRDID``
   * - Covariate adjustment
     - ✅ DR, IPW, Reg
     - ✅ DR, IPW, Reg
     - ✅ Varies
   * - Honest DiD
     - ✅ ``HonestDiD``
     - ``HonestDiD`` package
     - N/A
   * - Synthetic DiD
     - ✅ ``SyntheticDiD``
     - ``synthdid`` package
     - N/A
   * - Wild bootstrap
     - ✅ ``wild_bootstrap_se``
     - ``fwildclusterboot``
     - N/A

Package Correspondence
----------------------

R ``did`` Package → diff-diff
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The R ``did`` package by Callaway and Sant'Anna is the gold standard for
staggered DiD. Here's how to translate common operations:

**Basic estimation:**

.. code-block:: r

   # R (did package)
   library(did)
   out <- att_gt(
     yname = "Y",
     tname = "period",
     idname = "id",
     gname = "G",
     data = data
   )

.. code-block:: python

   # Python (diff-diff)
   from diff_diff import CallawaySantAnna

   cs = CallawaySantAnna()
   results = cs.fit(
       data,
       outcome='Y',
       time='period',
       unit='id',
       first_treat='G'
   )

**With covariates (doubly robust):**

.. code-block:: r

   # R
   out <- att_gt(
     yname = "Y", tname = "period",
     idname = "id", gname = "G",
     xformla = ~ X1 + X2,
     est_method = "dr",
     data = data
   )

.. code-block:: python

   # Python
   cs = CallawaySantAnna(estimation_method='dr')
   results = cs.fit(
       data,
       outcome='Y',
       time='period',
       unit='id',
       first_treat='G',
       covariates=['X1', 'X2']
   )

**Aggregations:**

.. code-block:: r

   # R
   agg_simple <- aggte(out, type = "simple")
   agg_dynamic <- aggte(out, type = "dynamic")
   agg_group <- aggte(out, type = "group")

.. code-block:: python

   # Python
   overall_att = results.att  # Simple aggregation
   event_study = results.aggregate('event_time')  # Dynamic
   by_group = results.aggregate('group')  # By cohort

R ``HonestDiD`` Package → diff-diff
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The HonestDiD package implements Rambachan & Roth (2023) sensitivity analysis:

**Relative magnitudes (ΔRM):**

.. code-block:: r

   # R
   library(HonestDiD)
   delta_rm_results <- createSensitivityResults_relativeMagnitudes(
     betahat = beta_hat,
     sigma = sigma,
     numPrePeriods = 4,
     numPostPeriods = 3,
     Mbarvec = seq(0, 2, by = 0.5)
   )

.. code-block:: python

   # Python
   from diff_diff import HonestDiD, DeltaRM

   honest = HonestDiD(delta=DeltaRM(M_bar=1.0))
   results = honest.fit(event_study_results)

   # Sensitivity analysis over M grid
   sensitivity = honest.sensitivity_analysis(
       event_study_results,
       M_grid=[0, 0.5, 1.0, 1.5, 2.0]
   )

**Smoothness restrictions (ΔSD):**

.. code-block:: r

   # R
   delta_sd_results <- createSensitivityResults(
     betahat = beta_hat,
     sigma = sigma,
     numPrePeriods = 4,
     numPostPeriods = 3,
     Mvec = seq(0, 0.1, by = 0.02)
   )

.. code-block:: python

   # Python
   from diff_diff import HonestDiD, DeltaSD

   honest = HonestDiD(delta=DeltaSD(M=0.05))
   results = honest.fit(event_study_results)

R ``synthdid`` Package → diff-diff
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The synthdid package implements Arkhangelsky et al. (2021):

.. code-block:: r

   # R
   library(synthdid)
   setup <- panel.matrices(data, unit = "unit", time = "time",
                           outcome = "Y", treatment = "treatment")
   tau.hat <- synthdid_estimate(setup$Y, setup$N0, setup$T0)

.. code-block:: python

   # Python
   from diff_diff import SyntheticDiD

   sdid = SyntheticDiD()
   results = sdid.fit(
       data,
       outcome='Y',
       unit='unit',
       time='time',
       treated='treatment',
       treatment_start=T0
   )

Key Differences
---------------

Design Philosophy
~~~~~~~~~~~~~~~~~

- **diff-diff**: sklearn-style API with ``fit()`` method, returning rich result objects
- **R packages**: Function-based, returning lists or S3/S4 objects

Inference
~~~~~~~~~

- **diff-diff**: Analytical SEs by default, wild bootstrap available
- **R did**: Multiplier bootstrap by default

Fixed Effects
~~~~~~~~~~~~~

- **diff-diff**: ``absorb`` parameter for high-dimensional FE (within transformation)
- **R fixest**: ``feols`` with ``|`` notation for absorbed FE

Output Format
~~~~~~~~~~~~~

diff-diff results have convenience methods:

.. code-block:: python

   results.summary()       # Print formatted table
   results.to_dict()       # Dictionary representation
   results.to_dataframe()  # pandas DataFrame

Feature Comparison Table
------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 15 15 15 15

   * - Feature
     - diff-diff
     - R did
     - R HonestDiD
     - R synthdid
   * - Basic 2x2 DiD
     - ✅
     - ✅
     - ❌
     - ❌
   * - TWFE
     - ✅
     - ❌
     - ❌
     - ❌
   * - Staggered DiD (CS)
     - ✅
     - ✅
     - ❌
     - ❌
   * - Covariate adjustment
     - ✅
     - ✅
     - ❌
     - ❌
   * - Doubly robust
     - ✅
     - ✅
     - ❌
     - ❌
   * - Group-time effects
     - ✅
     - ✅
     - ❌
     - ❌
   * - Event study
     - ✅
     - ✅
     - ✅
     - ❌
   * - Synthetic DiD
     - ✅
     - ❌
     - ❌
     - ✅
   * - Honest DiD (ΔRM)
     - ✅
     - ❌
     - ✅
     - ❌
   * - Honest DiD (ΔSD)
     - ✅
     - ❌
     - ✅
     - ❌
   * - Wild bootstrap
     - ✅
     - ❌
     - ❌
     - ❌
   * - Cluster-robust SE
     - ✅
     - ✅
     - ❌
     - ✅
   * - Placebo tests
     - ✅
     - ❌
     - ❌
     - ✅
   * - Parallel trends tests
     - ✅
     - ✅
     - ❌
     - ❌

Migration Tips
--------------

1. **Column names**: diff-diff uses string column names, similar to R packages

2. **Formula interface**: diff-diff supports R-style formulas for basic DiD:
   ``formula='y ~ treated * post'``

3. **Results access**: Use ``.att``, ``.se``, ``.ci`` instead of ``$att``, ``$se``

4. **Visualization**: ``plot_event_study()`` produces matplotlib figures similar
   to ``ggdid()`` output

5. **Missing data**: diff-diff requires complete data; use ``balance_panel()``
   or ``dropna()`` first
