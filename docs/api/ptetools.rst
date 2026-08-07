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
``did_rcs_attgt``; pass covariate names through ``covariates=`` for the DRDID
repeated-cross-section adjustment, or pass ``panel=False`` to ``pte`` for the
full RCS loop.
``covid_attgt`` provides the Callaway--Li levels-or-changes DRDID score.
Pass ``panel=False`` to ``pte`` for the repeated-cross-section main loop.
Full-history designs can use ``keep_all_untreated_subset`` or
``keep_all_pretreatment_subset``.
``setup_pte_basic``, ``pte_default``, and ``pte_attgt`` provide the standard
R-style convenience entry points.
Dynamic aggregation normalizes cohort weights within each event time and can
receive explicit ``cohort_weights``.
``PTEResults`` exposes ``summary()``, ``to_dict()``, and
``to_dataframe(level=...)`` for ATT(g,t), group, and dynamic surfaces.
Post-fit ``aggregate()`` results include influence-function standard errors
and normal-based confidence intervals when the influence functions are
available; bootstrap fits also expose the bootstrap distribution and
percentile ``overall_conf_int``.
Dynamic aggregation also accepts ``bstrap=True``, ``biters=`` and ``seed=`` for
multiplier-bootstrap pointwise and simultaneous bands.
``crit_val_checks`` validates simultaneous critical values before rendering
confidence bands.
Custom estimators can construct containers with ``group_time_att`` and
``aggte_obj``.
Use ``attgt_if`` for group-time results with an influence function and
``attgt_noif`` when the estimator only returns a point estimate.
``process_att_gt`` and ``attgt_pte_aggregations`` provide aggregation
dispatch aliases for custom group-time outputs.
``dose_obj`` and ``pte_dose_results`` provide a dose-response result surface.
``process_dose_gt`` combines per-group-time dose results into ATT(d) / ACRT(d)
curves and overall ATT/ACRT with multiplier-bootstrap standard errors;
``bspline_basis`` builds the splines2-compatible spline design used there.
``ggpte_cont`` is a compatibility wrapper around the project's dose-response
plotting API.
``ggpte`` is the event-study plotting wrapper for ``PTEResults``.
``panel_empirical_bootstrap`` and ``mboot2`` expose the two bootstrap engines
for custom workflows.
``pte_params``, ``pte_results``, and ``pte_emp_boot`` provide R-style aliases.
The generic ``pte`` loop accepts optional ``setup_pte_fun``, ``subset_fun``,
``attgt_fun``, and ``aggte_fun`` callbacks for custom ATT(g,t) estimators.
Callbacks use Python ``PTEParams``, ``TwoByTwoSubset``/``GTDataFrame``,
``ATTGTResult``, and ``PTEAggregateResult`` objects.
Quantile treatment effects are available through the QTT machinery:
``compute_pte`` runs the R ``compute.pte`` ``(g,t)`` loop, per-cell F0/F1
cumulative distribution functions are mixed with ``qtt_pte_aggregations`` (or
``qott_pte_aggregations`` for treatment-effect distributions), and
``qtt_empirical_bootstrap`` derives pointwise and simultaneous bands with
unit-level block bootstrap resampling. ``pte_qtt`` / ``PTEQTTResult`` hold the
resulting quantile curves, and ``block_boot_sample`` resamples a panel by unit.
``plot_qtt`` provides overall and dynamic matplotlib QTT plots.
Python-named S3-method counterparts are also available as
``autoplot_pte_results``, ``plot_pte_results``, ``autoplot_pte_qtt``,
``plot_pte_qtt``, ``autoplot_dose_obj``, and ``plot_dose_obj``.

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
   diff_diff.covid_attgt
   diff_diff.group_time_att
   diff_diff.aggte_obj
   diff_diff.process_att_gt
   diff_diff.attgt_pte_aggregations
   diff_diff.dose_obj
   diff_diff.pte_dose_results
   diff_diff.DoseResult
   diff_diff.ggpte_cont
   diff_diff.ggpte
   diff_diff.process_dose_gt
   diff_diff.bspline_basis
   diff_diff.mboot_se_and_crit
   diff_diff.panel_empirical_bootstrap
   diff_diff.mboot2
   diff_diff.pte_params
   diff_diff.pte_results
   diff_diff.pte_emp_boot
   diff_diff.setup_pte_basic
   diff_diff.pte_default
   diff_diff.pte_attgt
   diff_diff.attgt_if
   diff_diff.attgt_noif
   diff_diff.overall_weights
   diff_diff.pte_aggte
   diff_diff.pte
   diff_diff.PTEParams
   diff_diff.TwoByTwoSubset
   diff_diff.ATTGTResult
   diff_diff.PTEAggregateResult
   diff_diff.PTEResults
   diff_diff.compute_pte
   diff_diff.block_boot_sample
   diff_diff.qtt_pte_aggregations
   diff_diff.qott_pte_aggregations
   diff_diff.qtt_empirical_bootstrap
   diff_diff.pte_qtt
   diff_diff.plot_qtt
   diff_diff.autoplot_pte_results
   diff_diff.plot_pte_results
   diff_diff.autoplot_pte_qtt
   diff_diff.plot_pte_qtt
   diff_diff.autoplot_dose_obj
   diff_diff.plot_dose_obj
   diff_diff.PTEQTTResult
