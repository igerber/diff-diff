Staggered Adoption
==================

Estimators for staggered DiD designs where treatment is adopted at different times.

This module provides two main estimators for staggered adoption settings:

1. **Callaway-Sant'Anna (2021)**: Aggregates group-time 2x2 DiD comparisons
2. **Sun-Abraham (2021)**: Interaction-weighted regression approach

Running both provides a useful robustness check—when they agree, results are more credible.

.. module:: diff_diff.staggered

CallawaySantAnna
----------------

Callaway & Sant'Anna (2021) estimator for heterogeneous treatment timing.

.. autoclass:: diff_diff.CallawaySantAnna
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

   .. rubric:: Methods

   .. autosummary::

      ~CallawaySantAnna.fit
      ~CallawaySantAnna.get_params
      ~CallawaySantAnna.set_params

CallawaySantAnnaResults
-----------------------

Results container for Callaway-Sant'Anna estimation.

.. autoclass:: diff_diff.CallawaySantAnnaResults
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Methods

   .. autosummary::

      ~CallawaySantAnnaResults.aggregate
      ~CallawaySantAnnaResults.summary
      ~CallawaySantAnnaResults.to_dataframe

GroupTimeEffect
---------------

Container for individual group-time ATT(g,t) effects.

.. autoclass:: diff_diff.GroupTimeEffect
   :members:
   :undoc-members:
   :show-inheritance:

.. module:: diff_diff.sun_abraham

SunAbraham
----------

Sun & Abraham (2021) interaction-weighted estimator for staggered DiD.

This estimator provides event-study coefficients using a saturated regression
with cohort-by-relative-time interactions. It uses interaction-weighting to
aggregate cohort-specific effects into event study estimates.

**Key differences from Callaway-Sant'Anna:**

- Uses regression-based approach rather than 2x2 DiD comparisons
- Weights cohort-specific effects by share of each cohort in treated population
- Can be more efficient when treatment effects are homogeneous
- Running both provides a useful robustness check

**Reference:** Sun, L., & Abraham, S. (2021). Estimating dynamic treatment effects
in event studies with heterogeneous treatment effects. *Journal of Econometrics*, 225(2), 175-199.

.. autoclass:: diff_diff.SunAbraham
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

   .. rubric:: Methods

   .. autosummary::

      ~SunAbraham.fit
      ~SunAbraham.get_params
      ~SunAbraham.set_params
      ~SunAbraham.summary
      ~SunAbraham.print_summary

SunAbrahamResults
-----------------

Results container for Sun-Abraham estimation.

.. autoclass:: diff_diff.SunAbrahamResults
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Methods

   .. autosummary::

      ~SunAbrahamResults.summary
      ~SunAbrahamResults.print_summary
      ~SunAbrahamResults.to_dataframe

SABootstrapResults
------------------

Bootstrap inference results for Sun-Abraham estimation.

.. autoclass:: diff_diff.SABootstrapResults
   :members:
   :undoc-members:
   :show-inheritance:
