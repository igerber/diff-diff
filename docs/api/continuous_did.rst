Continuous Difference-in-Differences
=====================================

Continuous DiD estimator for dose-response curves with continuous treatment intensity.

This module implements the methodology from Callaway, Goodman-Bacon & Sant'Anna (2024),
"Difference-in-Differences with a Continuous Treatment" (NBER WP 32117), which:

1. **Estimates dose-response curves**: ATT(d) and ACRT(d) as functions of dose
2. **Computes summary parameters**: Overall ATT (binarized) and ACRT aggregated across doses
3. **Uses B-spline smoothing**: Flexible nonparametric estimation of dose-response functions
4. **Supports multiplier bootstrap**: Valid inference with proper standard errors and CIs

.. note::

   **Identification assumptions.** The dose-response curves ATT(d) and ACRT(d),
   as well as ATT\ :sup:`glob` and ACRT\ :sup:`glob`, require the **Strong Parallel
   Trends (SPT)** assumption — that there is no selection into dose groups based on
   treatment effects. Under the weaker standard Parallel Trends (PT) assumption,
   only the binarized ATT\ :sup:`loc` (``overall_att``) is identified; it equals
   ATT\ :sup:`glob` only when SPT holds. See Callaway, Goodman-Bacon & Sant'Anna
   (2024), Assumptions 1–2.

**When to use Continuous DiD:**

- Treatment varies in **intensity or dose** across units (not just binary on/off)
- You want to estimate how effects change with treatment dose
- Staggered adoption with heterogeneous dose levels
- You need the full dose-response curve, not just a single average effect

**Data requirements:**

- An **untreated group** (D = 0) must be present in the data
- A **balanced panel** is required (all units observed in all time periods)
- **Dose must be time-invariant** — each unit's dose is fixed across periods

**Reference:** Callaway, B., Goodman-Bacon, A., & Sant'Anna, P. H. C. (2024).
Difference-in-Differences with a Continuous Treatment. *NBER Working Paper* 32117.
`<https://www.nber.org/papers/w32117>`_

.. module:: diff_diff.continuous_did

ContinuousDiD
--------------

Main estimator class for Continuous Difference-in-Differences.

.. autoclass:: diff_diff.ContinuousDiD
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

   .. rubric:: Methods

   .. autosummary::

      ~ContinuousDiD.fit
      ~ContinuousDiD.get_params
      ~ContinuousDiD.set_params

ContinuousDiDResults
--------------------

Results container for Continuous DiD estimation.

.. autoclass:: diff_diff.continuous_did_results.ContinuousDiDResults
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Methods

   .. autosummary::

      ~ContinuousDiDResults.summary
      ~ContinuousDiDResults.print_summary
      ~ContinuousDiDResults.to_dataframe

DoseResponseCurve
-----------------

Dose-response curve container for ATT(d) or ACRT(d).

.. autoclass:: diff_diff.continuous_did_results.DoseResponseCurve
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Methods

   .. autosummary::

      ~DoseResponseCurve.to_dataframe

Example Usage
-------------

Basic usage::

    from diff_diff import ContinuousDiD, generate_continuous_did_data

    data = generate_continuous_did_data(n_units=200, seed=42)

    est = ContinuousDiD(n_bootstrap=199, seed=42)
    results = est.fit(data, outcome='outcome', unit='unit',
                      time='period', first_treat='first_treat',
                      dose='dose', aggregate='dose')
    results.print_summary()

Accessing dose-response curves::

    # ATT(d) dose-response curve as DataFrame
    att_df = results.dose_response_att.to_dataframe()
    print(att_df[['dose', 'effect', 'se', 'p_value']])

    # ACRT(d) dose-response curve
    acrt_df = results.dose_response_acrt.to_dataframe()

    # Overall summary parameters
    print(f"Overall ATT: {results.overall_att:.3f} (SE: {results.overall_att_se:.3f})")
    print(f"Overall ACRT: {results.overall_acrt:.3f} (SE: {results.overall_acrt_se:.3f})")

Event study aggregation::

    # Dynamic effects (binarized ATT by relative period)
    results_es = est.fit(data, outcome='outcome', unit='unit',
                         time='period', first_treat='first_treat',
                         dose='dose', aggregate='eventstudy')
    es_df = results_es.to_dataframe(level='event_study')

Comparison with CallawaySantAnna
--------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Feature
     - ContinuousDiD
     - CallawaySantAnna
   * - Treatment type
     - Continuous dose / intensity
     - Binary (treated / not treated)
   * - Target parameter
     - ATT\ :sup:`loc` (PT); ATT(d), ACRT(d), ATT\ :sup:`glob`, ACRT\ :sup:`glob` (SPT)
     - ATT(g,t), aggregated ATT
   * - Smoothing
     - B-spline basis for dose-response
     - None (nonparametric group-time)
   * - Dose-response curve
     - Yes (full curve with CIs)
     - No
   * - Bootstrap
     - Multiplier bootstrap (optional)
     - Multiplier bootstrap (optional)
   * - Control group
     - never_treated / not_yet_treated
     - never_treated / not_yet_treated
