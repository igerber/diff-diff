.. meta::
   :description: Measure marketing campaign impact with Python. Step-by-step guide to Difference-in-Differences for data scientists - from data to stakeholder-ready results.
   :keywords: python measure campaign impact, marketing DiD tutorial, python campaign lift, causal inference for data scientists, python geo experiment

Getting Started: Measuring Campaign Impact
==========================================

Your company ran a marketing campaign in 8 of 20 metro markets. Sales data is
available for all markets before and after the campaign. Leadership wants to know:
**did the campaign work, and by how much?**

This guide walks through the entire analysis - from data to a stakeholder-ready result.


What You'll Need
----------------

- Python 3.9+
- diff-diff installed (``pip install diff-diff``)
- About 15 minutes

.. code-block:: bash

   pip install diff-diff


Step 1: Set Up the Data
-----------------------

We will use diff-diff's data generator to create a realistic scenario: 20 markets
tracked over 12 months, with a campaign that launches in month 7 in 8 of the 20
markets. The true sales lift is 5 units per market.

.. code-block:: python

   from diff_diff import DifferenceInDifferences, generate_did_data

   data = generate_did_data(
       n_units=20,             # 20 metro markets
       n_periods=12,           # 12 months of data
       treatment_effect=5.0,   # true lift: 5 units per market
       treatment_fraction=0.4, # 8 of 20 markets got the campaign
       treatment_period=7,     # campaign launches in month 7
       seed=42,
   )

   print(data.head(10))
   print(f"\nMarkets: {data['unit'].nunique()}")
   print(f"Campaign markets: {data.loc[data['treated'] == 1, 'unit'].nunique()}")
   print(f"Control markets: {data.loc[data['treated'] == 0, 'unit'].nunique()}")

.. tip::

   **What the columns mean in business terms:**

   - ``unit`` = market ID (e.g., metro area)
   - ``period`` = month number (1-12)
   - ``treated`` = 1 if this market received the campaign, 0 for control
   - ``post`` = 1 for months after the campaign launched (month 7+)
   - ``outcome`` = the metric you are measuring (sales, signups, revenue, etc.)

   In your own data, these columns can have any name - you tell diff-diff which is
   which when you call ``fit()``.


Step 2: Look at the Data
-------------------------

Before running any analysis, plot the trends for campaign and control markets. This
visual check is the most important step.

.. code-block:: python

   # Requires matplotlib: pip install matplotlib
   import matplotlib.pyplot as plt

   trends = data.groupby(["period", "treated"])["outcome"].mean().unstack()
   trends.columns = ["Control Markets", "Campaign Markets"]

   fig, ax = plt.subplots(figsize=(10, 5))
   trends.plot(ax=ax, marker="o", linewidth=2)
   ax.axvline(x=6.5, color="gray", linestyle="--", label="Campaign Launch")
   ax.set_xlabel("Month")
   ax.set_ylabel("Sales")
   ax.set_title("Sales by Market Group Over Time")
   ax.legend()
   plt.tight_layout()
   plt.show()

What you are looking for: **before the campaign, the two lines should track roughly in
parallel.** If they diverge before the launch, something else is driving the difference
and DiD may give misleading results.

.. note::

   **Why this matters:** DiD assumes that without the campaign, both groups would have
   continued on the same trajectory. This is called the *parallel trends assumption*.
   It is the single most important condition for the analysis to be valid. You cannot
   prove it holds, but you can check whether it looks plausible.

.. tip::

   The plot requires ``matplotlib``, which is not a dependency of diff-diff. The
   analysis itself works without it - the plot just helps you sanity-check the data.


Step 3: Measure the Campaign Lift
---------------------------------

.. code-block:: python

   did = DifferenceInDifferences()
   results = did.fit(
       data,
       outcome="outcome",
       treatment="treated",
       post="post",
   )

   print(results.summary())

This prints a summary table with the estimate, standard error, confidence interval,
and p-value. (For a one-line summary, use ``print(results)`` instead.)

.. tip::

   **Reading the results in business terms:**

   - **ATT** = the estimated campaign lift (average increase in sales per market
     due to the campaign)
   - **Std. Err.** = how precisely we measured the lift (smaller is better)
   - **95% Confidence Interval** = the range we are 95% confident the true lift falls
     within. If it does not include zero, the effect is statistically significant.
   - **p-value** = the probability of seeing this result by chance if the campaign
     actually had no effect. Below 0.05 is conventionally considered significant.


Step 4: Check Whether the Result Is Trustworthy
------------------------------------------------

A statistically significant result is only meaningful if the underlying assumptions
hold. Two quick checks give you confidence (or flag problems).

Pre-campaign trend check
~~~~~~~~~~~~~~~~~~~~~~~~~

This tests whether campaign and control markets were trending at the same rate before
the launch.

.. code-block:: python

   from diff_diff import check_parallel_trends

   pt = check_parallel_trends(
       data,
       outcome="outcome",
       time="period",
       treatment_group="treated",
   )

   print(f"Pre-campaign trend difference: {pt['trend_difference']:.3f}")
   print(f"p-value: {pt['p_value']:.3f}")

   if pt["parallel_trends_plausible"]:
       print("Pre-campaign trends are consistent - the analysis is on solid ground.")
   else:
       print("Warning: trends diverge before the campaign. Investigate further.")

.. note::

   **What this checks:** Were the two groups trending at the same rate before the
   campaign? If yes, it supports (but does not prove) the assumption that they would
   have continued on the same trajectory. A non-significant p-value here is good news.

   **Academic term:** This is a *pre-trends test*. Note that passing this test does not
   guarantee the assumption holds - it is a necessary but not sufficient check.

Placebo test
~~~~~~~~~~~~

Run the same analysis on pre-campaign data only, using a fake launch date. If you
find a "significant" effect where none should exist, something is wrong with the method
or data.

.. code-block:: python

   # Use only pre-campaign data (months 1-6)
   pre_data = data[data["period"] < 7].copy()
   # Create a fake "launch" at month 4
   pre_data["placebo_post"] = (pre_data["period"] >= 4).astype(int)

   placebo = DifferenceInDifferences()
   placebo_results = placebo.fit(
       pre_data,
       outcome="outcome",
       treatment="treated",
       post="placebo_post",
   )

   print(f"Placebo lift: {placebo_results.att:.2f} (p = {placebo_results.p_value:.3f})")

   if placebo_results.p_value > 0.05:
       print("No spurious effect detected - the method is not picking up noise.")
   else:
       print("Warning: spurious effect found. Investigate the data for confounders.")

.. note::

   **What this checks:** If the method finds a "campaign effect" during a period when
   no campaign was running, it means something else is systematically different between
   the groups. This is called a *placebo test* or *falsification test*.


Step 5: Communicate the Result
------------------------------

Translate the statistical output into a stakeholder-ready statement.

.. code-block:: python

   r = results
   print(f"""
   Campaign Impact Summary
   =======================
   The campaign increased sales by {r.att:.1f} units per market
   (95% CI: {r.conf_int[0]:.1f} to {r.conf_int[1]:.1f}).

   This result is statistically significant (p = {r.p_value:.4f}).

   Validity checks:
   - Pre-campaign trends were consistent across groups (p = {pt['p_value']:.2f})
   - Placebo test detected no spurious effects
   """)

.. tip::

   **For your stakeholder report:**

   - Lead with the point estimate and confidence interval, not the p-value
   - Say "increased sales by X" not "the ATT is X"
   - Include "we verified that markets were trending similarly before the campaign"
   - Acknowledge uncertainty: "we are 95% confident the true lift is between A and B"
   - Separate statistical significance from practical significance: a statistically
     significant 0.1% lift may not justify the campaign spend


What If Your Campaign Rolled Out in Waves?
------------------------------------------

Many campaigns do not launch everywhere at once. If your campaign started in some
markets first and expanded later, you need a method designed for this - otherwise
the estimates can be biased.

.. code-block:: python

   from diff_diff import CallawaySantAnna, generate_staggered_data

   # Campaign launches: wave 1 at month 4, wave 2 at month 7
   data = generate_staggered_data(
       n_units=20, n_periods=12, cohort_periods=[4, 7],
       never_treated_frac=0.4, treatment_effect=5.0, seed=42,
   )

   cs = CallawaySantAnna()
   results = cs.fit(
       data, outcome="outcome", unit="unit",
       time="period", first_treat="first_treat",
   )

   print(f"Overall campaign lift: {results.overall_att:.1f}")
   print(results.summary())

.. note::

   **Why a different method?** When a campaign launches in waves, basic DiD can use
   already-active markets as "controls" for newly-launched ones - producing biased
   results. Callaway-Sant'Anna (2021) avoids this by comparing each wave only to
   markets that have not yet received the campaign.

   **Academic term:** This is a *staggered adoption* design. The method provides
   *ATT(g,t)* estimates for each wave at each time period, then aggregates them.

For the full staggered analysis workflow, see :doc:`practitioner_decision_tree`.


What If You Have Survey Data?
-----------------------------

If your outcome comes from a survey (brand awareness, NPS, customer satisfaction),
your data likely has a complex sampling design with strata, clusters, and weights.
Ignoring these makes your confidence intervals too narrow.

diff-diff handles this via :class:`~diff_diff.SurveyDesign` - pass it to any estimator's
``fit()`` method.

If your data is **individual-level microdata** - one row per respondent, with
sampling weights and strata/PSU columns (BRFSS, ACS, CPS, NHANES) - use
:func:`~diff_diff.aggregate_survey` first to roll it up to a geographic-period
panel. The helper computes design-based cell means and returns a pre-configured
``SurveyDesign`` with ``weight_type="pweight"`` (population weights) for the
second-stage fit. This second-stage design works with all survey-capable
estimators, including staggered estimators like
:class:`~diff_diff.CallawaySantAnna` and :class:`~diff_diff.ImputationDiD`.

.. code-block:: python

   from diff_diff import aggregate_survey, SurveyDesign, SunAbraham

   # 1. Describe the microdata's sampling design
   design = SurveyDesign(weights="finalwt", strata="strat", psu="psu")

   # 2. Roll up respondent records into a state-year panel
   panel, stage2 = aggregate_survey(
       microdata, by=["state", "year"],
       outcomes="brand_awareness", survey_design=design,
   )

   # 3. Add the campaign launch year per state, then fit a modern staggered
   #    estimator with the pre-configured second-stage SurveyDesign:
   # panel["first_treat"] = panel["state"].map(campaign_launch_year)  # NaN = control
   # results = SunAbraham().fit(
   #     panel, outcome="brand_awareness_mean",
   #     unit="state", time="year", first_treat="first_treat",
   #     survey_design=stage2,
   # )
   # results.print_summary()

For a complete walkthrough with brand funnel metrics and survey design corrections,
see `Tutorial 17: Brand Awareness Survey
<tutorials/17_brand_awareness_survey.ipynb>`_.


Next Steps
----------

- :doc:`practitioner_decision_tree` - Not sure which method fits? Match your scenario
- `Tutorial 17: Brand Awareness Survey <tutorials/17_brand_awareness_survey.ipynb>`_ -
  Full survey analysis with complex sampling design
- `Tutorial 18: Geo-Experiment Analysis with SyntheticDiD <tutorials/18_geo_experiments.ipynb>`_ -
  Geo-experiment walkthrough on a simulated DMA panel: SDiD fit, diagnostics, and stakeholder summary
- :doc:`choosing_estimator` - The complete academic estimator guide
- :doc:`api/index` - Full API reference
