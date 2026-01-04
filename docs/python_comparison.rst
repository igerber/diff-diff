Comparison with Python Packages
================================

This guide compares diff-diff with other Python packages for DiD analysis,
helping users understand the landscape and choose the right tool.

Overview
--------

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 15 15 15

   * - Feature
     - diff-diff
     - pyfixest
     - differences
     - CausalPy
     - linearmodels
   * - Basic DiD
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅ (manual)
   * - TWFE
     - ✅
     - ✅
     - ❌
     - ❌
     - ✅
   * - Callaway-Sant'Anna
     - ✅
     - ❌
     - ✅
     - ❌
     - ❌
   * - Sun-Abraham
     - ❌
     - ✅
     - ❌
     - ❌
     - ❌
   * - Synthetic DiD
     - ✅
     - ❌
     - ❌
     - ✅ (SC only)
     - ❌
   * - Honest DiD
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌
   * - Wild bootstrap
     - ✅
     - ✅
     - ❌
     - ❌
     - ❌

Package Summaries
-----------------

pyfixest
~~~~~~~~

`pyfixest <https://github.com/py-econometrics/pyfixest>`_ is inspired by R's
``fixest`` package and excels at fast fixed effects estimation.

**Strengths:**

- Very fast for high-dimensional fixed effects (JIT-compiled)
- Sun-Abraham event study estimator
- Gardner's did2s two-stage estimator
- Local projections (Dube et al. 2023)
- Publication-ready tables

**Gaps:**

- No Callaway-Sant'Anna estimator
- No sensitivity analysis (Honest DiD)
- No synthetic DiD

differences
~~~~~~~~~~~

`differences <https://pypi.org/project/differences/>`_ provides a pure
Callaway-Sant'Anna implementation.

**Strengths:**

- Faithful implementation of Callaway & Sant'Anna (2021)
- Multi-valued treatment support
- Triple difference estimation
- Handles unbalanced panels

**Gaps:**

- No sensitivity analysis
- No synthetic DiD
- Limited inference options (no wild bootstrap)

CausalPy
~~~~~~~~

`CausalPy <https://github.com/pymc-labs/CausalPy>`_ takes a Bayesian approach
using PyMC for uncertainty quantification.

**Strengths:**

- Bayesian credible intervals
- Synthetic control methods
- Nice built-in visualizations
- Interrupted time series

**Gaps:**

- Limited to simple 2x2 DiD designs
- No staggered treatment support
- No event study methods

linearmodels
~~~~~~~~~~~~

`linearmodels <https://github.com/bashtage/linearmodels>`_ provides panel data
econometrics, serving as a building block for DiD.

**Strengths:**

- Solid PanelOLS implementation
- Two-way fixed effects
- Various standard error options
- statsmodels-compatible

**Gaps:**

- No DiD-specific methods (manual implementation required)
- No staggered treatment handling
- No diagnostics or sensitivity analysis

Why diff-diff?
--------------

diff-diff fills critical gaps in the Python DiD ecosystem:

1. **Only Python library with Honest DiD sensitivity analysis**

   Reviewers increasingly expect sensitivity analysis showing robustness to
   parallel trends violations. diff-diff is the only Python package implementing
   Rambachan & Roth (2023).

2. **Unified toolkit**

   Combines Callaway-Sant'Anna, Synthetic DiD, TWFE, and Honest DiD in one
   package with a consistent API.

3. **Built-in diagnostics**

   Placebo tests, parallel trends tests, Goodman-Bacon decomposition, and
   sensitivity analysis out of the box.

4. **sklearn-style API**

   Familiar ``fit()`` interface with rich result objects.

Feature Comparison Table
------------------------

.. list-table::
   :header-rows: 1
   :widths: 35 13 13 13 13 13

   * - Feature
     - diff-diff
     - pyfixest
     - differences
     - CausalPy
     - linearmodels
   * - Basic 2x2 DiD
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
   * - Two-way fixed effects
     - ✅
     - ✅
     - ❌
     - ❌
     - ✅
   * - Staggered DiD (CS)
     - ✅
     - ❌
     - ✅
     - ❌
     - ❌
   * - Sun-Abraham estimator
     - ❌
     - ✅
     - ❌
     - ❌
     - ❌
   * - Gardner's did2s
     - ❌
     - ✅
     - ❌
     - ❌
     - ❌
   * - Local projections
     - ❌
     - ✅
     - ❌
     - ❌
     - ❌
   * - Synthetic DiD
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌
   * - Synthetic control
     - ❌
     - ❌
     - ❌
     - ✅
     - ❌
   * - Covariate adjustment
     - ✅
     - ✅
     - ✅
     - Limited
     - ✅
   * - Doubly robust
     - ✅
     - ❌
     - ✅
     - ❌
     - ❌
   * - IPW
     - ✅
     - ❌
     - ✅
     - ❌
     - ❌
   * - Group-time effects
     - ✅
     - ❌
     - ✅
     - ❌
     - ❌
   * - Event study aggregation
     - ✅
     - ✅
     - ✅
     - ❌
     - ❌
   * - Honest DiD (ΔRM)
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌
   * - Honest DiD (ΔSD)
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌
   * - Sensitivity analysis
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌
   * - Wild bootstrap
     - ✅
     - ✅
     - ❌
     - ❌
     - ❌
   * - Cluster-robust SE
     - ✅
     - ✅
     - ✅
     - ❌
     - ✅
   * - Placebo tests
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌
   * - Parallel trends tests
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌
   * - Bacon decomposition
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌
   * - Event study plots
     - ✅
     - ✅
     - ✅
     - ✅
     - ❌
   * - Bayesian inference
     - ❌
     - ❌
     - ❌
     - ✅
     - ❌

Code Comparison
---------------

Basic DiD
~~~~~~~~~

.. code-block:: python

   # diff-diff
   from diff_diff import DifferenceInDifferences

   did = DifferenceInDifferences()
   results = did.fit(data, outcome='y', treated='treated', post='post')
   print(results.summary())

.. code-block:: python

   # pyfixest
   import pyfixest as pf

   fit = pf.feols("y ~ treated:post | unit + time", data=data)
   fit.summary()

.. code-block:: python

   # linearmodels
   from linearmodels.panel import PanelOLS

   mod = PanelOLS.from_formula(
       'y ~ treated:post + EntityEffects + TimeEffects',
       data=data.set_index(['unit', 'time'])
   )
   results = mod.fit(cov_type='clustered', cluster_entity=True)

Staggered DiD (Callaway-Sant'Anna)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # diff-diff
   from diff_diff import CallawaySantAnna

   cs = CallawaySantAnna(estimation_method='dr')
   results = cs.fit(
       data,
       outcome='y',
       unit='unit',
       time='time',
       first_treat='first_treat',
       covariates=['x1', 'x2']
   )
   event_study = results.aggregate('event_time')

.. code-block:: python

   # differences
   from differences import ATTgt

   att = ATTgt(
       data=data,
       outcome='y',
       cohort='first_treat',
       time='time',
       strata='unit'
   )
   att.fit()
   att.aggregate('event')

Sensitivity Analysis
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # diff-diff (only Python option)
   from diff_diff import HonestDiD, DeltaRM, plot_sensitivity

   honest = HonestDiD(delta=DeltaRM(M_bar=1.0))
   results = honest.fit(event_study_results)

   # Sensitivity over M grid
   sensitivity = honest.sensitivity_analysis(
       event_study_results,
       M_grid=[0, 0.5, 1.0, 1.5, 2.0]
   )
   plot_sensitivity(sensitivity)

   # No equivalent in other Python packages

When to Use Each Package
------------------------

**Use diff-diff when:**

- You need sensitivity analysis (Honest DiD)
- You want Callaway-Sant'Anna with doubly robust estimation
- You need built-in diagnostics (placebo tests, parallel trends)
- You prefer a unified sklearn-style API

**Use pyfixest when:**

- Speed is critical (high-dimensional fixed effects)
- You need Sun-Abraham or did2s estimators
- You want R's fixest-style syntax
- You need local projections

**Use differences when:**

- You need multi-valued treatment effects
- You need triple difference estimation
- You want the most faithful CS implementation

**Use CausalPy when:**

- You want Bayesian uncertainty quantification
- You're doing synthetic control (not synthetic DiD)
- You prefer PyMC-based modeling

**Use linearmodels when:**

- You're doing general panel econometrics beyond DiD
- You need statsmodels compatibility
- You're implementing DiD manually with full control
