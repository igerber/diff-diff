Visualization
=============

Plotting functions for DiD results visualization.

.. module:: diff_diff.visualization

plot_event_study
----------------

Create publication-ready event study coefficient plots.

.. autofunction:: diff_diff.plot_event_study

Example
~~~~~~~

.. code-block:: python

   from diff_diff import MultiPeriodDiD, plot_event_study

   # Fit event study model
   model = MultiPeriodDiD(reference_period=-1)
   results = model.fit(data, outcome='y', treated='treated',
                       time='period', unit='unit_id', treatment_start=5)

   # Create plot
   fig = plot_event_study(results)
   fig.savefig('event_study.png', dpi=300, bbox_inches='tight')

plot_group_effects
------------------

Visualize treatment effects by cohort.

.. autofunction:: diff_diff.plot_group_effects

Example
~~~~~~~

.. code-block:: python

   from diff_diff import CallawaySantAnna, plot_group_effects

   cs = CallawaySantAnna()
   results = cs.fit(data, outcome='y', unit='unit_id',
                    time='period', first_treat='first_treat')

   # Plot effects by treatment cohort
   fig = plot_group_effects(results)

plot_sensitivity
----------------

Plot Honest DiD sensitivity analysis results.

.. autofunction:: diff_diff.plot_sensitivity

Example
~~~~~~~

.. code-block:: python

   from diff_diff import HonestDiD, DeltaRM, plot_sensitivity

   honest = HonestDiD(delta=DeltaRM(M_bar=1.0))
   sensitivity = honest.sensitivity_analysis(
       results,
       M_grid=[0, 0.5, 1.0, 1.5, 2.0]
   )

   fig = plot_sensitivity(sensitivity)

plot_honest_event_study
-----------------------

Event study plot with honest confidence intervals.

.. autofunction:: diff_diff.plot_honest_event_study

Example
~~~~~~~

.. code-block:: python

   from diff_diff import HonestDiD, DeltaRM, plot_honest_event_study

   honest = HonestDiD(delta=DeltaRM(M_bar=1.0))
   bounds = honest.fit(event_study_results)

   fig = plot_honest_event_study(event_study_results, bounds)
