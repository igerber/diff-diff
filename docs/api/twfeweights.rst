TWFE ATT(g,t) Weights
=====================

Compatibility utilities ported from the no-covariate portion of the R
``twfeweights`` package.  They decompose an ATT(g,t) table into the weights
used by a two-way fixed-effects regression, Callaway--Sant'Anna's overall ATT,
or the simple overall ATT.

The functions accept a DataFrame with ``group``, ``time`` and ``attgt``
columns plus the original panel DataFrame.  The panel must contain a cohort
column, passed through ``treatment_group``.

The motivation and interpretation of these diagnostics follow Caetano and
Callaway (2026), especially the discussion of hidden linearity bias and
implicit regression weights.  See :doc:`../references` for the full citation.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   diff_diff.twfe_weights
   diff_diff.attO_weights
   diff_diff.att_simple_weights
   diff_diff.MPWeightsResult
   diff_diff.ggtwfeweights
   diff_diff.effective_sample_size
   diff_diff.pooled_sd
   diff_diff.log_ratio_sd
   diff_diff.frac_treated_extreme
   diff_diff.TwoPeriodCovariatesResult
   diff_diff.two_period_reg_weights
   diff_diff.two_period_aipw_weights
