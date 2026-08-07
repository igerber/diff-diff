R ``ptetools`` Compatibility
============================

The ``diff_diff.ptetools`` layer provides small, composable primitives for
group-time treatment effects. It is intended for users porting custom
estimators from R ``ptetools`` or building an estimator around a particular
ATT(g,t) score. For the standard staggered-adoption workflow, use
:class:`~diff_diff.CallawaySantAnna` directly.

R-to-Python map
---------------

The main public R functions have Python counterparts:

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - R ``ptetools``
     - Python
     - Purpose
   * - ``setup_pte`` / ``setup_pte_basic``
     - ``setup_pte`` / ``setup_pte_basic``
     - Validate a panel and define estimable cells
   * - ``two_by_two_subset``
     - ``two_by_two_subset``
     - Construct one group-time comparison
   * - ``did_attgt``
     - ``did_attgt``
     - Estimate a two-period ATT(g,t)
   * - ``covid_attgt``
     - ``covid_attgt``
     - Callaway--Li levels or changes DRDID score
   * - ``pte``
     - ``pte``
     - Run the generic group-time loop
   * - ``pte_aggte``
     - ``pte_aggte``
     - Group or dynamic aggregation

Basic group-time workflow
-------------------------

The generic ``pte`` function expects a panel with an outcome, treatment
cohort, period, and unit identifier. Cohort ``0`` denotes never-treated
units.

Custom ATT(g,t) estimators can replace the default subset, cell estimator, or
aggregation step with ``subset_fun``, ``attgt_fun``, and ``aggte_fun``. These
callbacks are Python-native equivalents of the corresponding R ``pte``
extension points.

.. code-block:: python

   from diff_diff import ggpte, pte

   results = pte(
       data,
       yname="outcome",
       gname="first_treat",
       tname="period",
       idname="unit",
       covariates=["income", "population"],
   )

   print(results.summary())
   att_gt = results.to_dataframe()                 # ATT(g,t) rows
   dynamic = results.to_dataframe("dynamic")      # event-time rows
   dynamic_result = results.aggregate("dynamic")
   print(dynamic_result.to_dict())
   ax = ggpte(results, show=False)

   # Optional multiplier-bootstrap dynamic bands.
   dynamic_boot = results.aggregate("dynamic", bstrap=True, biters=500, seed=42)
   print(dynamic_boot.to_dataframe())

``aggregate("dynamic")`` uses the retained unit-level influence functions to
compute the aggregate standard error and normal-based confidence interval.
``to_dataframe("group")`` and ``to_dataframe("dynamic")`` return the
corresponding post-fit aggregate view; the default ``to_dataframe()`` remains
the ATT(g,t) table.

Callaway--Li / DRDID cell score
-------------------------------

``covid_attgt`` reuses the same DRDID-validated doubly-robust panel core used
by ``CallawaySantAnna(estimation_method="dr")``. The input is a two-period
``GTDataFrame`` with ``name`` equal to ``"pre"`` or ``"post"`` and ``D`` as
the treatment indicator.

.. code-block:: python

   from diff_diff import covid_attgt, gt_data_frame

   gt_data = gt_data_frame(two_period_data)

   # Levels relative to a zero untreated baseline, matching d_outcome=False.
   levels = covid_attgt(
       gt_data,
       covariates=["age", "prior_outcome"],
       d_covariates=["employment"],
   )

   # First-difference outcome, matching d_outcome=True.
   changes = covid_attgt(
       gt_data,
       covariates=["age", "prior_outcome"],
       d_covariates=["employment"],
       d_outcome=True,
   )

The returned ``ATTGTResult.inf_func`` uses the Python convention
``phi = psi / n``. This is the scale consumed by the aggregation and bootstrap
helpers; R's DRDID object exposes the unnormalized ``psi`` representation.

Quantile and dose-response outputs
----------------------------------

The QTT/QoTT and dose-response surfaces are post-fit result containers:

.. code-block:: python

   qtt = pte_qtt(...)
   qtt_table = qtt.to_dataframe()
   ax = plot_qtt(qtt, type="overall", show=False)
   ax = plot_qtt(qtt, type="dynamic", plot_probs=[0.5], show=False)

   dose = process_dose_gt(gt_results, ptep)
   ax = ggpte_cont(dose, type="att", show=False)
   ax = ggpte_cont(dose, type="acrt", show=False)

``ggpte`` and ``ggpte_cont`` return the project's standard matplotlib axes by
default. Pass ``backend="plotly"`` to ``ggpte_cont`` for an interactive Plotly
figure.

Further reference
-----------------

See :doc:`api/ptetools` for the complete callable and result-container
reference, and :doc:`r_comparison` for the broader R/Python comparison.
