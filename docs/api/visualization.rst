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
   model = MultiPeriodDiD()
   results = model.fit(data, outcome='y', treatment='treated',
                       time='period', unit='unit_id', reference_period=2)

   # Create plot
   ax = plot_event_study(results)
   ax.figure.savefig('event_study.png', dpi=300, bbox_inches='tight')

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
   ax = plot_group_effects(results)

plot_sensitivity
----------------

Plot Honest DiD sensitivity analysis results.

.. autofunction:: diff_diff.plot_sensitivity

Example
~~~~~~~

.. code-block:: python

   from diff_diff import HonestDiD, plot_sensitivity

   honest = HonestDiD(method='relative_magnitude', M=1.0)
   sensitivity = honest.sensitivity_analysis(
       results,
       M_grid=[0, 0.5, 1.0, 1.5, 2.0]
   )

   ax = plot_sensitivity(sensitivity)

plot_honest_event_study
-----------------------

Event study plot with honest confidence intervals.

.. autofunction:: diff_diff.plot_honest_event_study

Example
~~~~~~~

.. code-block:: python

   from diff_diff import HonestDiD, plot_honest_event_study

   honest = HonestDiD(method='relative_magnitude', M=1.0)
   bounds = honest.fit(event_study_results)

   ax = plot_honest_event_study(bounds)

plot_synth_weights
------------------

Visualize synthetic control unit or time weights.

.. autofunction:: diff_diff.plot_synth_weights

Example
~~~~~~~

.. code-block:: python

   from diff_diff import plot_synth_weights

   # Plot from a weights dictionary
   weights = {"unit_A": 0.40, "unit_B": 0.30, "unit_C": 0.20, "unit_D": 0.10}
   ax = plot_synth_weights(weights=weights)

   # Or from SyntheticDiD results:
   # ax = plot_synth_weights(results)
   # ax = plot_synth_weights(results, weight_type='time')

plot_staircase
--------------

Visualize treatment adoption timing in staggered designs.

.. autofunction:: diff_diff.plot_staircase

Example
~~~~~~~

.. code-block:: python

   from diff_diff import CallawaySantAnna, plot_staircase

   cs = CallawaySantAnna()
   results = cs.fit(data, outcome='y', unit='unit_id',
                    time='period', first_treat='first_treat')

   # Staircase plot from results
   ax = plot_staircase(results)

   # Or from raw panel data
   ax = plot_staircase(data=data, unit='unit_id', time='period',
                       first_treat='first_treat')

plot_dose_response
------------------

Visualize dose-response curves from continuous DiD.

.. autofunction:: diff_diff.plot_dose_response

Example
~~~~~~~

.. code-block:: python

   import pandas as pd
   from diff_diff import plot_dose_response

   # Plot from a DataFrame with dose-response data
   dr_data = pd.DataFrame({
       'dose': [1, 2, 3, 4, 5],
       'effect': [0.1, 0.3, 0.5, 0.4, 0.3],
       'se': [0.05, 0.06, 0.07, 0.08, 0.09],
   })
   ax = plot_dose_response(data=dr_data)

   # Or from ContinuousDiD results:
   # ax = plot_dose_response(results, target='att')
   # ax = plot_dose_response(results, target='acrt')

plot_group_time_heatmap
-----------------------

Heatmap of group-time treatment effects.

.. autofunction:: diff_diff.plot_group_time_heatmap

Example
~~~~~~~

.. code-block:: python

   from diff_diff import CallawaySantAnna, plot_group_time_heatmap

   cs = CallawaySantAnna()
   results = cs.fit(data, outcome='y', unit='unit_id',
                    time='period', first_treat='first_treat')

   # Heatmap of ATT(g,t)
   ax = plot_group_time_heatmap(results)

   # Grey out non-significant cells
   ax = plot_group_time_heatmap(results, mask_insignificant=True)

Plotly Backend
--------------

All visualization functions support an interactive plotly backend via the
``backend`` parameter:

.. code-block:: python

   # Interactive event study
   fig = plot_event_study(results, backend='plotly')

   # Interactive dose-response curve
   fig = plot_dose_response(results, backend='plotly')

Install plotly with: ``pip install diff-diff[plotly]``
