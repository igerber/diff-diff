DiagnosticReport
================

``DiagnosticReport`` orchestrates the library's existing diagnostic
functions (parallel trends, pre-trends power, HonestDiD sensitivity,
Goodman-Bacon, design-effect, EPV, heterogeneity, and estimator-native
checks for SyntheticDiD and TROP) into a single report with a stable
AI-legible schema.

Construction is free; accessing ``applicable_checks`` may derive the
fit's post-fit event-study surface once (a view or kit recompute via
``results.aggregate('event_study')``, cached for the report's
lifetime, and only when the raw ``event_study_effects`` field is
absent); ``run_all()`` triggers the full check computation and caches.
A second call to ``to_dict()`` or ``summary()`` reuses the cached
result.

Methodology deviations (no traffic-light gates, opt-in placebo
battery, estimator-native diagnostic routing, power-aware phrasing
threshold) are documented in :doc:`/methodology/REPORTING`.

The schema carries a top-level ``target_parameter`` block
(experimental) naming what the headline scalar represents per
estimator. See the "Target parameter" section of
:doc:`/methodology/REPORTING`
for the per-estimator dispatch and schema shape.

Data-dependent checks (2x2 parallel trends on simple DiD,
Goodman-Bacon decomposition on staggered estimators, the EfficientDiD
Hausman PT-All vs PT-Post pretest) require the raw panel + column
names. Pass ``data``, ``outcome``, ``treatment``, ``unit``, ``time``,
and/or ``first_treat`` and they feed the runners. Without these
kwargs, those specific checks are skipped with an explicit reason
while the rest of the battery still runs.

For survey-weighted fits (any result carrying
``survey_metadata``) pass the original ``SurveyDesign`` via
``survey_design=<design>``. It is threaded through to
``BaconDecomposition.fit`` for a fit-faithful Goodman-Bacon replay. When
``survey_metadata`` is set but ``survey_design`` is not supplied,
Bacon is skipped with an explicit reason so the report never emits
an unweighted decomposition for a design that differs from the
estimate; alternatively supply
``precomputed={'bacon': <BaconDecompositionResults>}`` with a
survey-aware result.

The simple 2x2 parallel-trends helper has no survey-aware variant
and is skipped unconditionally on a survey-backed ``DiDResults``
regardless of ``survey_design`` — the helper cannot consume the
design even when it is available. Supply
``precomputed={'parallel_trends': <dict>}`` with a survey-aware
pretest result to opt in.

Example
-------

.. code-block:: python

   from diff_diff import CallawaySantAnna, DiagnosticReport

   cs = CallawaySantAnna(base_period="universal").fit(
       df, outcome="outcome", unit="unit", time="period",
       first_treat="first_treat",
   )
   # The event-study-gated checks (parallel trends, pre-trends power,
   # sensitivity) derive the surface internally via the result's
   # post-fit aggregate('event_study') when needed.
   dr = DiagnosticReport(
       cs,
       data=df,
       outcome="outcome",
       unit="unit",
       time="period",
       first_treat="first_treat",
   )
   print(dr.summary())
   dr.to_dataframe()  # one row per check

API
---

.. autoclass:: diff_diff.DiagnosticReport
   :no-index:
   :members:
   :show-inheritance:

.. autoclass:: diff_diff.DiagnosticReportResults
   :no-index:
   :members:
   :show-inheritance:

.. autodata:: diff_diff.DIAGNOSTIC_REPORT_SCHEMA_VERSION

DurationDiD native hazard diagnostics
-------------------------------------

For ``DurationDiDResults``, the stored fixed-anchor hazard pretest is extracted
without raw data, refitting, or recomputing diagnostics. The
``estimator_native_diagnostics`` section has outer ``status='ran'`` for successful
extraction; its nested ``pretrend_test.status`` determines availability and
``reject=None`` denotes an unavailable test. Its confidence level is the fit's
``alpha``, independently of report-level phrasing. Non-rejection never establishes
identification or adequate power. Effect and diagnostic bootstrap validity are
independent; inspect estimation/inference statuses, reasons and support warnings.

``summary()`` (including on the returned ``DiagnosticReportResults``) and
``full_report()`` describe the stored hazard decision at its fitted simultaneous
confidence level. An unavailable pretest includes its reasons and explicitly has
no rejection decision. The native row in ``to_dataframe()`` retains extraction
``status='ran'``: its ``headline`` is the stored hazard-test p-value when available,
otherwise missing, with the availability reasons in ``reason``. These views never
recompute the diagnostic or replace its fitted confidence level with report alpha.
For invalid counterfactual curves, the narrative directs readers to
``survival_curve`` and specification comparisons using ``method`` and ``fit_periods``.

All generic ``precomputed`` overrides (parallel_trends, sensitivity,
pretrends_power, bacon) are rejected for DurationDiD. Its hazard contrasts are
not pre-treatment outcome ATTs and do not admit generic HonestDiD/PreTrendsPower.

A self-contained example:

.. code-block:: python

   import pandas as pd
   from diff_diff import DurationDiD, DiagnosticReport

   rows = []
   for group, counts in enumerate(([96, 80, 64, 48, 32, 16], [90, 75, 60, 45, 20, 8])):
       for person in range(400):
           for date, survivors in enumerate(counts):
               rows.append((group * 400 + person, date, group, int(person >= 4 * survivors)))
   panel = pd.DataFrame(rows, columns=["person", "date", "group", "absorbed"])
   fitted = DurationDiD(n_bootstrap=19, seed=33).fit(
       panel, "absorbed", "group", "person", "date", post_periods=[4, 5]
   )
   native = DiagnosticReport(fitted).to_dict()["estimator_native_diagnostics"]
   assert native["estimator"] == "DurationDiD"
   print(fitted.pretrend_test().summary())
