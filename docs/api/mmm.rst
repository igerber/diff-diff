MMM Calibration Export
======================

Assemble Marketing Mix Model (MMM) calibration inputs from experiment results.

.. module:: diff_diff.mmm

Overview
--------

.. seealso::

   Tutorials 29 (:doc:`../tutorials/29_mmm_calibration_pymc`) and 30
   (:doc:`../tutorials/30_mmm_calibration_meridian`) run both hand-offs
   end-to-end against the real frameworks - a lift-test calibration in
   PyMC-Marketing and an ROI-prior calibration in Meridian - including the
   with/without-calibration posterior comparison.

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

**Explicit numbers in, or the pinned aggregation contract in - validated out.**
Reconciling an experiment's estimate to a calibration input needs context diff-diff
cannot always see - the target MMM's row granularity (per-geo vs national), its
time window, and the outcome's scale (additive levels vs a log/rate/share). The
default route stays fully explicit: the caller supplies the already-scoped
incremental outcome and its standard error (the numbers read off a fitted result's
``summary()``, aggregated to the population and window one MMM row represents).
Alternatively both exporters accept ``aggregation_result=`` - the pinned
:class:`~diff_diff.AggregationResult` returned by ``results.aggregate('simple')``,
``results.aggregate('group')`` (each with ``scale=``), or
``results.aggregate('total')`` (already the total; no scale) - deriving
``effect = att * scale`` and ``se = se * scale`` per row; ``scale="auto"`` (reading
the container's own treated-observation count) is honored only for ImputationDiD
and TwoStageDiD fits, and acknowledges assumptions the container cannot verify
(additive-level outcome, unweighted fit, fully identified effects - see the
docstrings). diff-diff does only what it can verify: assemble the exact schema,
enforce each consumer's guards, convert to the lognormal parameterization (parity
with Google's closed form), pool, and emit snippets. It is pure numpy/pandas and
imports no MMM package.

to_pymc_marketing_lift_test
---------------------------

Assemble a lift-test DataFrame for ``pymc_marketing.mmm.MMM.add_lift_test_measurements``.

.. autofunction:: diff_diff.to_pymc_marketing_lift_test

Example
~~~~~~~

.. code-block:: python

   import numpy as np
   import pandas as pd

   from diff_diff import SyntheticDiD, to_pymc_marketing_lift_test

   # Single-treated-geo experiment (US-CA): TV spend raised there vs control
   # geos. With exactly ONE treated geo, the SDID ATT is that geo's per-week
   # lift, so labelling the row with its coordinate is sound. (For MULTIPLE
   # treated geos the pooled ATT is an average - do not assign it to one geo;
   # export each geo's own effect, or omit the geo dim and match aggregate
   # spend to an aggregate lift.)
   rng = np.random.default_rng(42)
   geos = [f"g{i}" for i in range(9)] + ["US-CA"]
   panel = pd.DataFrame(
       {
           "geo": g,
           "week": w,
           "treated": int(g == "US-CA"),  # block treatment: constant per unit
           "revenue": 100.0 + 2.0 * w + rng.normal(0.0, 1.0)
           + (8.0 if g == "US-CA" and w >= 8 else 0.0),
       }
       for g in geos
       for w in range(12)
   )
   result = SyntheticDiD().fit(panel, outcome='revenue', treatment='treated',
                               unit='geo', time='week',
                               post_periods=list(range(8, 12)))

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

Deriving totals from a fitted aggregation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The primary route (3.10) is ``results.aggregate('total')`` - the
estimator-owned total incremental outcome on CallawaySantAnna, EfficientDiD,
ImputationDiD, and TwoStageDiD (panel non-survey fits): its single row is
already ``C x overall`` over the estimator's finite-masked complete-case
support, so the exporter takes the container alone and rejects any ``scale``.
For overall-total exports this supersedes ``scale="auto"``; ``"auto"`` remains
the route for ImputationDiD/TwoStageDiD ``group``-level (per-cohort)
containers. Either way the outcome must be in additive levels (see the
docstrings):

.. code-block:: python

   from diff_diff import ImputationDiD, to_pymc_marketing_lift_test
   from diff_diff.prep import generate_staggered_data

   data = generate_staggered_data(n_units=60, n_periods=6, seed=11)
   res = ImputationDiD().fit(data, outcome='outcome', unit='unit',
                             time='period', first_treat='first_treat')

   df_lift = to_pymc_marketing_lift_test(
       channel='tv',
       x=50_000.0,
       delta_x=20_000.0,
       aggregation_result=res.aggregate('total'),  # already the total; no scale
   )
   # delta_y == the total row's att, sigma == its se; the usual guards apply.
   # Scaled-container alternatives: aggregation_result=res.aggregate('simple')
   # (one overall row) or res.aggregate('group') (per-cohort rows) with
   # scale='auto' - ImputationDiD's n IS the treated unit-periods there, at
   # the cost of the documented raw-support caveat.

to_meridian_roi_prior
---------------------

Build a Meridian lognormal ``roi_m``/``mroi_m`` prior from scoped experiment result(s).

.. autofunction:: diff_diff.to_meridian_roi_prior

Example
~~~~~~~

.. code-block:: python

   from diff_diff import DifferenceInDifferences, to_meridian_roi_prior
   from diff_diff.prep import generate_did_data

   panel = generate_did_data(n_units=60, n_periods=2, treatment_effect=5.0,
                             treatment_period=1, seed=7)
   result = DifferenceInDifferences().fit(panel, outcome='outcome',
                                          treatment='treated', post='post')

   # On the explicit route the caller aggregates the ATT to a total incremental
   # outcome over the treated population and window (e.g. att x treated units x
   # post periods for an unweighted additive fit) and supplies its SE; the
   # aggregation-container route below derives the numbers instead.
   prior = to_meridian_roi_prior(
       incremental_outcome=180_000.0,     # total incremental revenue
       incremental_outcome_se=45_000.0,   # its standard error
       spend=200_000.0,                   # total channel spend over the window
       parameter='roi_m',                 # or 'mroi_m' for a marginal experiment
       se_widening=1.5,                   # optional transferability skepticism
   )
   print(prior.roi_mean, prior.roi_sd)

Scoping the prior to the experiment window
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``to_code()`` requires the prior's time scope. Build the boolean
``(n_media_times, n_media_channels)`` mask with
:func:`~diff_diff.meridian_calibration_mask` and pass it directly - the array is
serialized into the generated snippet (a string expression or
``full_model_window=True`` also work). Note Meridian's own guidance: its
configure-model guide states that ``roi_calibration_period`` "is not generally
recommended because calibrating the ROI of a specific time period does not
necessarily improve estimation of the overall ROI" - prefer
``full_model_window=True`` when the experiment evidence reasonably transfers to
the full window, and reserve the mask for evidence genuinely specific to a
narrower period:

.. code-block:: python

   import pandas as pd

   from diff_diff import meridian_calibration_mask, to_meridian_roi_prior

   prior = to_meridian_roi_prior(
       incremental_outcome=180_000.0,
       incremental_outcome_se=45_000.0,
       spend=200_000.0,
   )

   # The MMM's own coordinates: time labels in model order, channels in
   # InputData order. window=(start, end) is inclusive; pass a list for
   # explicit (possibly non-contiguous) time labels instead.
   media_times = pd.date_range('2023-09-04', periods=52, freq='W-MON')
   mask = meridian_calibration_mask(
       media_times=media_times,
       media_channels=['search', 'tv'],
       channel='tv',
       window=('2024-01-15', '2024-03-04'),
   )
   print(prior.to_code(channel='tv', media_channels=['search', 'tv'],
                       roi_calibration_period=mask))
   # The snippet rebuilds the same mask (all-True base; the experiment
   # channel's column carries only the window - other channels keep ALL
   # periods, Meridian's documented convention) and passes it to
   # ModelSpec(roi_calibration_period=...). roi_m priors only: Meridian
   # rejects the mask for mroi_m (use full_model_window=True there).

Deriving totals from a fitted aggregation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Estimators outside the ``scale="auto"`` allowlist reach the exporter through
``results.aggregate('total')`` too - where supported, that container needs no
scale at all. On the routings totals do not support (repeated-cross-section,
declared ``survey_design=``, divergent bare-``cluster=`` masses), the caller
supplies a numeric scale instead - a caller-defined estimand, needed because
those containers' ``n`` does not count treated unit-periods (CallawaySantAnna's
``simple`` container, for example, counts treated *and* control units):

.. code-block:: python

   from diff_diff import CallawaySantAnna, to_meridian_roi_prior
   from diff_diff.prep import generate_staggered_data

   data = generate_staggered_data(n_units=60, n_periods=6, seed=11)
   cs = CallawaySantAnna().fit(data, outcome='outcome', unit='unit',
                               time='period', first_treat='first_treat')

   prior = to_meridian_roi_prior(
       aggregation_result=cs.aggregate('simple'),
       scale=132.0,        # caller-derived treated unit-periods for THEIR scoping
       spend=200_000.0,
   )
   # incremental_outcome == att * 132.0, and its SE == se * 132.0.

meridian_calibration_mask
-------------------------

Build the boolean ``roi_calibration_period`` mask from the MMM's own coordinates.

.. autofunction:: diff_diff.meridian_calibration_mask

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
- Google Meridian, "Set the ROI calibration period" (the
  ``roi_calibration_period`` mask shape/semantics contract):
  https://developers.google.com/meridian/docs/user-guide/configure-model
- Zhou, G., Choe, Y., & Hetrakul, C. (2023). Calibrated MMM better predicts true
  ROAS. Meta Marketing Science.
