MMM Calibration Export
======================

Assemble Marketing Mix Model (MMM) calibration inputs from experiment results.

.. module:: diff_diff.mmm

Overview
--------

MMM practitioners calibrate their models against experimental evidence. The two
dominant Python MMM frameworks consume that evidence in different shapes:

1. **PyMC-Marketing** (and prophetverse, which mirrors the same schema) ingests a
   lift-test DataFrame - columns ``channel``, optional model dims (e.g. ``geo``),
   ``x``, ``delta_x``, ``delta_y``, ``sigma`` - via
   ``MMM.add_lift_test_measurements``.

2. **Google Meridian** calibrates through lognormal priors: ``roi_m`` for the return
   on a channel's full spend (a zero-spend / full-holdout estimand) and ``mroi_m``
   for a marginal return. The point estimate maps to the prior mean and its standard
   error to the prior standard deviation, converted to lognormal ``(mu, sigma)`` with
   Google's closed form.

**Explicit in, validated out.** These functions do not rescale a result's headline
ATT. Reconciling an experiment's estimate to a calibration input needs context
diff-diff cannot see - the target MMM's row granularity (per-geo vs national), its
time window, and the outcome's scale (additive levels vs a log/rate/share) - so the
caller supplies the already-scoped incremental outcome and its standard error (the
numbers read off a fitted result's ``summary()``, aggregated to the population and
window one MMM row represents). diff-diff does only what it can verify: assemble the
exact schema, enforce each consumer's guards, convert to the lognormal
parameterization (parity with Google's closed form), pool, and emit snippets. It is
pure numpy/pandas and imports no MMM package.

to_pymc_marketing_lift_test
---------------------------

Assemble a lift-test DataFrame for ``pymc_marketing.mmm.MMM.add_lift_test_measurements``.

.. autofunction:: diff_diff.to_pymc_marketing_lift_test

Example
~~~~~~~

.. code-block:: python

   from diff_diff import SyntheticDiD, to_pymc_marketing_lift_test

   # Single-treated-geo experiment (US-CA): TV spend raised there vs control
   # geos. With exactly ONE treated geo, the SDID ATT is that geo's per-week
   # lift, so labelling the row with its coordinate is sound. (For MULTIPLE
   # treated geos the pooled ATT is an average - do not assign it to one geo;
   # export each geo's own effect, or omit the geo dim and match aggregate
   # spend to an aggregate lift.)
   result = SyntheticDiD().fit(panel, outcome='revenue', treatment='treated',
                               unit='geo', time='week')

   df_lift = to_pymc_marketing_lift_test(
       channel='tv',
       x=50_000.0,           # baseline weekly TV spend in US-CA
       delta_x=20_000.0,     # spend increase during the test
       delta_y=result.att,   # US-CA's per-week lift (one treated geo)
       sigma=result.se,      # its standard error
       dims={'geo': 'US-CA'},
   )
   # -> columns [channel, geo, x, delta_x, delta_y, sigma], ready for
   #    MMM.add_lift_test_measurements(df_lift)

to_meridian_roi_prior
---------------------

Build a Meridian lognormal ``roi_m``/``mroi_m`` prior from scoped experiment result(s).

.. autofunction:: diff_diff.to_meridian_roi_prior

Example
~~~~~~~

.. code-block:: python

   from diff_diff import DifferenceInDifferences, to_meridian_roi_prior

   result = DifferenceInDifferences().fit(panel, outcome='revenue',
                                          treatment='treated', post='post')

   # Caller aggregates the ATT to a total incremental outcome over the treated
   # population and window (e.g. att x treated units x post periods) and supplies
   # its SE - diff-diff does not rescale the headline effect.
   prior = to_meridian_roi_prior(
       incremental_outcome=180_000.0,     # total incremental revenue
       incremental_outcome_se=45_000.0,   # its standard error
       spend=200_000.0,                   # total channel spend over the window
       parameter='roi_m',                 # or 'mroi_m' for a marginal experiment
       se_widening=1.5,                   # optional transferability skepticism
   )
   print(prior.roi_mean, prior.roi_sd)

   # Channel- and time-scoped snippet: roi_m is per-channel, and the prior was
   # estimated on the experiment window.
   print(prior.to_code(channel='tv', media_channels=['search', 'tv'],
                       roi_calibration_period='experiment_window_mask'))

MeridianROIPrior
----------------

Result container for the Meridian exporter.

.. autoclass:: diff_diff.MeridianROIPrior
   :no-index:
   :members:
   :undoc-members:

References
----------

- PyMC-Marketing lift-test calibration:
  https://www.pymc-marketing.io/en/stable/notebooks/mmm/mmm_lift_test.html
- Google Meridian, "Set custom prior distributions using past experiments":
  https://developers.google.com/meridian/docs/advanced-modeling/set-custom-priors-past-experiments
- Google Meridian, ROI/mROI/contribution parameterizations:
  https://developers.google.com/meridian/docs/advanced-modeling/roi-mroi-contribution-parameterizations
- Zhou, G., Choe, Y., & Hetrakul, C. (2023). Calibrated MMM better predicts true
  ROAS. Meta Marketing Science.
