DiagnosticReport
================

``DiagnosticReport`` orchestrates the library's existing diagnostic
functions (parallel trends, pre-trends power, HonestDiD sensitivity,
Goodman-Bacon, design-effect, EPV, heterogeneity, and estimator-native
checks for SyntheticDiD and TROP) into a single report with a stable
AI-legible schema.

Construction is free; ``run_all()`` triggers the compute and caches.
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
       first_treat="first_treat", aggregate="event_study",
   )
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
