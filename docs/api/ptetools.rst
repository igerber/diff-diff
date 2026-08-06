Panel Treatment-Effects Primitives
===================================

The ``ptetools`` compatibility layer exposes composable building blocks for
custom panel treatment-effect estimators.  Use ``setup_pte`` to validate and
describe a panel, ``two_by_two_subset`` to create a group-time comparison, and
``did_attgt`` to estimate an unadjusted two-period ATT.  Custom estimators can
return ``ATTGTResult`` objects and aggregate group-time effects with
``pte_aggte``. The ``pte`` wrapper runs the complete unadjusted group-time
loop.
Pass pre-period column names through ``covariates=`` to use the conditional
AIPW path in ``did_attgt`` and ``pte``.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   diff_diff.setup_pte
   diff_diff.two_by_two_subset
   diff_diff.gt_data_frame
   diff_diff.did_attgt
   diff_diff.attgt_if
   diff_diff.overall_weights
   diff_diff.pte_aggte
   diff_diff.pte
   diff_diff.PTEParams
   diff_diff.TwoByTwoSubset
   diff_diff.ATTGTResult
   diff_diff.PTEAggregateResult
   diff_diff.PTEResults
