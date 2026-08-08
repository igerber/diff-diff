Staggered Adoption
==================

Estimators for staggered DiD designs where treatment is adopted at different times.

This module provides three estimators for staggered adoption settings:

1. **Callaway-Sant'Anna (2021)**: Aggregates group-time 2x2 DiD comparisons
2. **Sun-Abraham (2021)**: Interaction-weighted regression approach
3. **Ortiz-Villavicencio & Sant'Anna (2025)**: Staggered triple-difference (DDD) with group-time ATT

Running CS and SA together provides a useful robustness check - when they agree, results are more credible.

.. module:: diff_diff.staggered

CallawaySantAnna
----------------

Callaway & Sant'Anna (2021) estimator for heterogeneous treatment timing.

.. autoclass:: diff_diff.CallawaySantAnna
   :no-index:
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
   :no-index:
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
   :no-index:
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
   :no-index:
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
   :no-index:
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
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

StaggeredTripleDifference
-------------------------

Ortiz-Villavicencio & Sant'Anna (2025) staggered triple-difference (DDD) estimator
with group-time ATT identification under heterogeneous treatment timing.

.. deprecated:: 3.9
   Removed in 4.0 (ledger row M-013). Use
   :class:`~diff_diff.TripleDifference` with
   ``fit(..., unit=, time=, first_treat=, partition=)``, which runs the same
   engine. ``eligibility=`` is named ``partition=`` there, and ``control_group``
   takes the underscored values ``"not_yet_treated"``/``"never_treated"``. The
   ``SDDD`` alias is deprecated with the class.

.. autoclass:: diff_diff.StaggeredTripleDifference
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

StaggeredTripleDiffResults
--------------------------

Results container for ``StaggeredTripleDifference`` estimation.

.. autoclass:: diff_diff.StaggeredTripleDiffResults
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:
