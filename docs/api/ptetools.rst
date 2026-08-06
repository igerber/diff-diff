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
AIPW path in ``did_attgt`` and ``pte``. Set ``bstrap=True`` to use the
unit-level empirical bootstrap with a reproducible ``seed``.
Repeated-cross-section designs use ``two_by_two_rcs_subset`` and
``did_rcs_attgt``.
Full-history designs can use ``keep_all_untreated_subset`` or
``keep_all_pretreatment_subset``.
``setup_pte_basic``, ``pte_default``, and ``pte_attgt`` provide the standard
R-style convenience entry points.
Dynamic aggregation normalizes cohort weights within each event time and can
receive explicit ``cohort_weights``.
``PTEResults`` exposes ``summary()``, ``to_dict()``, and the bootstrap
distribution and percentile ``overall_conf_int`` when empirical bootstrap
inference is requested.
``crit_val_checks`` validates simultaneous critical values before rendering
confidence bands.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   diff_diff.setup_pte
   diff_diff.two_by_two_subset
   diff_diff.two_by_two_rcs_subset
   diff_diff.keep_all_untreated_subset
   diff_diff.keep_all_pretreatment_subset
   diff_diff.gt_data_frame
   diff_diff.did_attgt
   diff_diff.did_rcs_attgt
   diff_diff.setup_pte_basic
   diff_diff.pte_default
   diff_diff.pte_attgt
   diff_diff.attgt_if
   diff_diff.overall_weights
   diff_diff.pte_aggte
   diff_diff.pte
   diff_diff.PTEParams
   diff_diff.TwoByTwoSubset
   diff_diff.ATTGTResult
   diff_diff.PTEAggregateResult
   diff_diff.PTEResults
