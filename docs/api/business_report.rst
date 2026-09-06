BusinessReport
==============

``BusinessReport`` wraps any fitted diff-diff result object and produces
stakeholder-ready output:

- ``summary()`` — a short paragraph block suitable for an email or Slack.
- ``full_report()`` — a structured multi-section markdown report.
- ``to_dict()`` — a stable AI-legible structured schema (single source
  of truth; prose renders from this dict).

By default, BusinessReport constructs an internal ``DiagnosticReport``
to surface pre-trends, sensitivity, and other validity checks as part
of the narrative. Pass ``auto_diagnostics=False`` to skip this, or
``diagnostics=<DiagnosticReport>`` to supply an explicit one.

Pre-computed diagnostics can be forwarded directly to the auto-
constructed ``DiagnosticReport`` via
``precomputed={'parallel_trends': ...}``,
``precomputed={'sensitivity': ...}``,
``precomputed={'pretrends_power': ...}``, or
``precomputed={'bacon': ...}`` — same keys as
``DiagnosticReport(precomputed=...)``. DR validates keys and rejects
estimator-incompatible entries.

Data-dependent checks (2x2 parallel trends on simple DiD,
Goodman-Bacon decomposition on staggered estimators, the EfficientDiD
Hausman PT-All vs PT-Post pretest) require the raw panel + column
names. Pass ``data``, ``outcome``, ``treatment``, ``unit``, ``time``,
and/or ``first_treat`` to ``BusinessReport`` and they are forwarded
to the auto-constructed ``DiagnosticReport``. Without these kwargs,
those specific checks are skipped with an explicit reason while the
rest of the report still renders.

For survey-weighted fits (any result carrying
``survey_metadata``) pass the original ``SurveyDesign`` via
``survey_design=<design>``. It is threaded through to
``BaconDecomposition.fit`` for a fit-faithful Goodman-Bacon replay. When
``survey_metadata`` is set but ``survey_design`` is not supplied,
Bacon is skipped with an explicit reason so the report never emits
an unweighted decomposition for a design that differs from the
estimate. The simple 2x2 parallel-trends helper has no survey-aware
variant and is skipped unconditionally on a survey-backed
``DiDResults`` regardless of ``survey_design``; supply
``precomputed={'parallel_trends': ...}`` with a survey-aware
pretest to opt in.

Methodology deviations (no traffic-light gates, pre-trends verdict
thresholds, power-aware phrasing, unit-translation policy, schema
stability) are documented in :doc:`/methodology/REPORTING`.

The schema carries a top-level ``target_parameter`` block
(experimental) naming what the headline scalar represents per
estimator — simple ATT, event-study average, DID_M, DID_1,
cost-benefit delta, dose-response aggregate, factor-model-adjusted ATT,
etc. For the dCDH dynamic branch with ``trends_linear=True`` and
``L_max>=2``, the scalar is intentionally NaN and
``aggregation`` is ``"no_scalar_headline"`` with
``headline_attribute`` set to ``None``. Agents should dispatch on
this case and inspect the headline ``reason`` field, which
distinguishes the populated-surface subcase (per-horizon table
available on ``linear_trends_effects``) from the empty-surface
subcase (no horizons survived estimation; re-fit with a larger
``L_max`` or with ``trends_linear=False``). See the "Target
parameter" section of :doc:`/methodology/REPORTING`
for the full per-estimator dispatch table and schema shape.

Example
-------

.. code-block:: python

   from diff_diff import CallawaySantAnna, BusinessReport, generate_staggered_data

   # Staggered-rollout loyalty program across stores
   df = generate_staggered_data(
       n_units=60, n_periods=10, cohort_periods=[4, 7],
       never_treated_frac=0.3, treatment_effect=5.0, seed=42,
   ).rename(columns={"unit": "store", "outcome": "revenue"})

   cs = CallawaySantAnna(base_period="universal").fit(
       df, outcome="revenue", unit="store", time="period",
       first_treat="first_treat",
   )
   # The auto-constructed DiagnosticReport derives the event-study
   # surface internally via post-fit aggregate('event_study') when the
   # pre-trends checks need it.
   report = BusinessReport(
       cs,
       outcome_label="Revenue per store",
       outcome_unit="$",
       business_question="Did the loyalty program lift revenue?",
       treatment_label="the loyalty program",
       # Optional: panel + column names so auto diagnostics can run the
       # data-dependent checks (2x2 PT, Goodman-Bacon, EfficientDiD
       # Hausman). Without these the auto path still runs and just
       # skips those checks.
       data=df,
       outcome="revenue",
       unit="store",
       time="period",
       first_treat="first_treat",
   )
   print(report.summary())

API
---

.. autoclass:: diff_diff.BusinessReport
   :no-index:
   :members:
   :show-inheritance:

.. autoclass:: diff_diff.BusinessContext
   :no-index:
   :members:
   :show-inheritance:

.. autodata:: diff_diff.BUSINESS_REPORT_SCHEMA_VERSION

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

All generic ``precomputed`` overrides (parallel_trends, sensitivity,
pretrends_power, bacon) are rejected for DurationDiD. Its hazard contrasts are
not pre-treatment outcome ATTs and do not admit generic HonestDiD/PreTrendsPower.

BusinessReport preserves this payload in ``robustness.estimator_native`` and
renders hazard-test availability or rejection in summary and full reports.
``honest_did_results`` is also rejected. A report alpha override preserves the
fitted bootstrap confidence intervals and identifies their actual confidence
level; it cannot reconstruct intervals from the reported SE.

An explicit ``diagnostics=`` argument, whether a live ``DiagnosticReport`` or a
detached ``DiagnosticReportResults``, must name DurationDiD and contain its
matching stored native hazard diagnostic and availability metadata. Foreign
reports, altered native payloads and computed generic diagnostic sections raise
``ValueError``. Construct the diagnostic report from the fitted results being
reported. Use ``pretrend_test()`` and comparisons of ``method`` and ``fit_periods``
for the supported diagnostic and specification workflow.

``auto_diagnostics=False`` skips automatic DiagnosticReport construction, but
fit-level caveats remain visible: invalid counterfactual curves, unavailable
inference, failed bootstrap counts, support warnings and unavailable hazard
pretests. When at most three treated people are present, the caveat describes
pooled individual-bootstrap support and reliability, not large-cluster
asymptotics, synthetic weighting or exact permutation inference. See
:doc:`duration_did` and :doc:`diagnostic_report` for the fitting example.

The short summary groups support warnings by count and summarizes unavailable
inference families. Every warning and reason remains in ``to_dict()`` and
``full_report()``, including details for each date.
