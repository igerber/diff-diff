.. meta::
   :description: Compare diff-diff with R packages for DiD analysis. Migration guide from R did, fixest, synthdid, and HonestDiD to Python with side-by-side code examples.
   :keywords: R did package python alternative, fixest python, synthdid python, R to python DiD, econometrics R vs python

R Comparison
============

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

Argument mapping
----------------

diff-diff deliberately ships **no argument aliases** for the R spellings. Where the
library owns a concept it uses its own name; where a name is the field's language it
uses that. This table is the translation layer that replaces aliasing — it is the
answer to "why isn't ``first_treat`` spelled ``gname``".

``Where it lives`` names the callable that accepts the argument, since not every R
argument maps onto ``fit()``.

.. list-table::
   :header-rows: 1
   :widths: 12 20 24 26 18

   * - R package
     - R argument
     - diff-diff equivalent
     - Where it lives
     - Notes
   * - ``did``
     - ``yname``
     - ``outcome``
     - ``CallawaySantAnna.fit``
     -
   * - ``did``
     - ``tname``
     - ``time``
     - ``CallawaySantAnna.fit``
     -
   * - ``did``
     - ``idname``
     - ``unit``
     - ``CallawaySantAnna.fit``
     -
   * - ``did``
     - ``gname``
     - ``first_treat``
     - ``CallawaySantAnna.fit``
     - First treated period; ``0`` marks never-treated
   * - ``did``
     - ``xformla``
     - ``covariates``
     - ``CallawaySantAnna.fit``
     - A list of column names, not a formula
   * - ``did``
     - ``est_method``
     - ``estimation_method``
     - ``CallawaySantAnna``
     - Constructor, not ``fit()``
   * - ``did``
     - ``aggte(type=)``
     - ``type``
     - ``CallawaySantAnnaResults.aggregate``
     - Post-fit, not a fit argument. ``"dynamic"`` becomes ``"event_study"``
   * - ``HonestDiD``
     - ``Mbarvec`` / ``Mvec``
     - ``M_grid``
     - ``HonestDiD.sensitivity_analysis``
     -
   * - ``HonestDiD``
     - ``betahat`` / ``sigma``
     - carried by the fitted results
     - results object
     - Pass the results object itself; there is no coefficient/vcov argument
   * - ``HonestDiD``
     - ``numPrePeriods`` / ``numPostPeriods``
     - inferred from the results
     - results object
     - Derived from the event-study surface, never passed
   * - ``synthdid``
     - ``Y`` (outcome matrix)
     - ``outcome``
     - ``SyntheticDiD.fit``
     - Long-format column name, not a matrix
   * - ``synthdid``
     - ``N0`` / ``T0`` (control/pre counts)
     - ``treatment`` + ``post_periods``
     - ``SyntheticDiD.fit``
     - Not a 1:1 mapping: a time-invariant treatment indicator plus the post-period list replaces the block structure

``fixest``, ``DIDmultiplegtDYN`` and ``DIDHAD`` are discussed on this page in prose
rather than with paired examples, so they are deliberately absent here rather than
mapped from memory. See :doc:`migration-4.0` for the 4.0 renames, which change several
of the diff-diff spellings above.

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

   # Python (R's aggte() has two counterparts: the fit-time aggregate= shown here,
   # deprecated in 3.9, and post-fit results.aggregate(type=), which supersedes it)
   results = cs.fit(data, outcome='Y', time='period', unit='id',
                    first_treat='G', aggregate='all')
   overall_att = results.overall_att  # Simple aggregation
   event_study = results.event_study_effects  # Dynamic
   by_group = results.group_effects  # By cohort

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
   from diff_diff import HonestDiD

   honest = HonestDiD(method='relative_magnitude', M=1.0)
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
   from diff_diff import HonestDiD

   honest = HonestDiD(method='smoothness', M=0.05)
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

   # SyntheticDiD requires a time-invariant ever-treated indicator
   data['ever_treated'] = data.groupby('unit')['treatment'].transform('max')

   # Derive post-treatment periods from treatment timing
   post_periods = sorted(data.loc[data['treatment'] == 1, 'time'].unique())

   sdid = SyntheticDiD()
   results = sdid.fit(
       data,
       outcome='Y',
       unit='unit',
       time='time',
       treatment='ever_treated',
       post_periods=post_periods
   )

Heterogeneous Adoption (HAD)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When every unit is treated at the post period (universal-rollout policies,
industry-wide regime changes) but treatment intensity varies across units,
the standard R workhorses (``did``, ``fixest``, ``synthdid``,
``DIDmultiplegtDYN``) assume an untreated comparison group exists and do
not apply. The dedicated R package ``DIDHAD`` (de Chaisemartin et al.,
August 2025) covers the QUG case (Design 1', ``d_lower = 0``) from the
same arXiv paper.

``diff-diff`` ships :class:`~diff_diff.HeterogeneousAdoptionDiD`, which
implements de Chaisemartin, Ciccia, D'Haultfoeuille and Knau (2026,
arXiv:2405.04465v6) and adds two surfaces beyond the QUG-focused R
package: Design 1 (no QUG, ``d_lower > 0``, targets ``WAS_{d_lower}`` under
Assumption 6 or sign-only under Assumption 5), and survey-design
integration via Binder (1983) Taylor-series linearization (sampling weights
+ optional strata / PSU / FPC). The diagnostic battery
:func:`~diff_diff.did_had_pretest_workflow` surfaces violations of the HAD
identification assumptions (the design path is auto-detected separately by
:meth:`HeterogeneousAdoptionDiD.fit` from the dose support).

.. code-block:: python

   import numpy as np
   import pandas as pd
   from diff_diff import HeterogeneousAdoptionDiD

   # Build a HAD-shape panel: D=0 in pre-periods (t < F), D > 0 only at F+.
   rng = np.random.default_rng(42)
   G, F, T = 200, 4, 5
   doses = rng.beta(0.5, 1.0, size=G)
   rows = []
   for g in range(G):
       for t in range(1, T + 1):
           y = (rng.normal()
                + (doses[g] + doses[g] ** 2) * (t >= F)
                + rng.normal(0, 0.5))
           d = doses[g] if t >= F else 0.0
           rows.append({'unit': g, 'period': t, 'y': y, 'dose': d})
   had_data = pd.DataFrame(rows)

   est = HeterogeneousAdoptionDiD()
   results = est.fit(had_data, outcome='y', unit='unit',
                     time='period', dose='dose')

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
   * - Bacon decomposition
     - ✅
     - ❌
     - ❌
     - ❌
   * - Sun-Abraham
     - ✅
     - ❌
     - ❌
     - ❌
   * - Imputation DiD
     - ✅
     - ❌
     - ❌
     - ❌
   * - Two-Stage DiD (did2s)
     - ✅
     - ❌
     - ❌
     - ❌
   * - Stacked DiD
     - ✅
     - ❌
     - ❌
     - ❌
   * - Continuous DiD
     - ✅
     - ✅
     - ❌
     - ❌
   * - Triple Difference (DDD)
     - ✅
     - ❌
     - ❌
     - ❌
   * - TROP
     - ✅
     - ❌
     - ❌
     - ❌
   * - Efficient DiD
     - ✅
     - ❌
     - ❌
     - ❌
   * - Heterogeneous adoption (HAD)
     - ✅
     - ❌
     - ❌
     - ❌

.. note::

   R equivalents for estimators not covered by the ``did``, ``HonestDiD``, or
   ``synthdid`` packages: Sun-Abraham is available via ``fixest::sunab()``;
   Imputation DiD via the ``didimputation`` package; Two-Stage DiD via the
   ``did2s`` package; Bacon Decomposition via the ``bacondecomp`` package;
   Stacked DiD requires manual implementation or the ``stackedev`` package;
   Continuous DiD is available via the ``did`` package continuous extension;
   Triple Difference requires manual implementation in R.
   Changes-in-Changes and QDiD are available via the ``qte`` package
   (``qte::CiC()`` / ``qte::QDiD()``, the diff-diff parity target -
   including covariates: diff-diff's ``covariates=`` ports qte's
   ``xformla`` branch, a quantreg-based conditional CiC).
   TROP and Efficient DiD have no direct R equivalents.
   HeterogeneousAdoptionDiD (dCDH 2026) overlaps with the dedicated R
   package ``DIDHAD`` (de Chaisemartin et al., 2025), which covers the
   QUG case (Design 1'); diff-diff additionally covers Design 1 (no QUG,
   ``WAS_{d_lower}``) and survey-design integration via Binder TSL.

Migration Tips
--------------

1. **Column names**: diff-diff uses string column names, similar to R packages

2. **Formula interface**: diff-diff supports R-style formulas for basic DiD:
   ``formula='y ~ treated * post'``

3. **Results access**: Use ``.att``, ``.se``, ``.conf_int`` instead of ``$att``, ``$se``

4. **Visualization**: ``plot_event_study()`` produces matplotlib figures similar
   to ``ggdid()`` output

5. **Missing data**: diff-diff requires complete data; use ``balance_panel()``
   or ``dropna()`` first

6. **Heterogeneous Adoption (HAD)**: If you need surfaces the R ``DIDHAD``
   package does not cover - Design 1 (no QUG, ``WAS_{d_lower}``) or
   survey-design integration - reach for
   :class:`~diff_diff.HeterogeneousAdoptionDiD`. See the
   `Heterogeneous Adoption (HAD)`_ section above for the migration pattern.
