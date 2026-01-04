Staggered Adoption
==================

Estimators for staggered DiD designs where treatment is adopted at different times.

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
