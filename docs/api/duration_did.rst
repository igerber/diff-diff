Duration DiD
============

``DurationDiD`` estimates cumulative absorption effects for two groups of
individuals observed on a complete, balanced panel with common treatment timing.
The outcome is 0 before absorption and 1 afterward. Treatment is a fixed group
indicator; ``post_periods`` separately declares the treated observation suffix.
Keep baseline-absorbed people in the panel. Positive ATT increases absorption.

Identification and supported designs
------------------------------------

Identification assumes no anticipation, unaffected controls, absorbing outcomes,
a fixed population, common timing and either common dynamics (an additive gap
between untreated hazards) or proportional untreated hazards. These are hazard
restrictions, not parallel trends in the mean binary outcome. Independence
across individuals is an additional bootstrap inference assumption.

At least two pre-periods and one post-period are required; the stored hazard
pretest needs three pre-periods. Numeric, Timestamp and Timedelta clocks must
have equal actual spacing. Numeric spacing comparisons use relative tolerance
1e-9 and absolute tolerance 1e-12 times the first spacing; datetime/timedelta
spacings must agree exactly. The estimator works on normalized observation
intervals and records ``time_origin`` / ``time_step``.

Covariates, staggered adoption, censoring/dropout, repeated cross-sections,
survey/sampling weights and higher-level clustering are unsupported. No
``survey_design`` or ``cluster`` parameter exists. ``time_weights`` weights
calibration dates, not individuals. Do not drop incomplete histories or fill
outcomes to claim the fixed-population assumptions have been established.

Calibration and point estimates
--------------------------------

``fit_periods`` selects dates after the original baseline and before treatment;
it never changes the diagnostic window or post horizon. ``time_weights`` keys
must match explicit selected dates, or all default eligible dates. Weights must
be finite and nonnegative with positive total. Zero-weight moments are removed
before arithmetic. Positive weights that underflow during normalization raise
instead of silently changing support. The effective positive-weight support remains fixed in every
bootstrap draw. Unsupported explicit positive-weight moments raise; default
exclusions carry reasons.

For whole-group survival S, R=-log(S), baseline increments D, normalized elapsed
durations d, treated group 1 and control group 0:

.. math::

   \hat c_{CD}=\sum_t w_t(D_{1t}-D_{0t})/d_t,\qquad
   \hat R^0_{1t}=\hat R_{1b}+D_{0t}+d_t\hat c_{CD}.

.. math::

   \hat c_{PH}=\sum_t w_t D_{1t}/D_{0t},\qquad
   \hat R^0_{1t}=\hat R_{1b}+\hat c_{PH}D_{0t}.

.. math::

   \widehat{ATT}_t=\exp(-\hat R^0_{1t})-S_{1t}.

The CD coefficient is a hazard gap per normalized observation interval; the PH
coefficient is a dimensionless hazard ratio. PH uses the mean of ratios and
requires a positive fitted coefficient; it is not the inverted printed slope.
The headline averages all declared post-period absorption ATTs uniformly.

Zero factual treated post-survival is valid. Required baseline, calibration and
control extrapolation log domains must remain defined. Invalid counterfactual
probabilities or increasing survival (including the factual treated last-pre boundary) suppress
canonical causal output while preserving raw extrapolations. Curves are never
clipped, monotonized or silently shortened. Roundoff tolerance is 1e-12.
This conservative library gate can fail from sampling noise under correctly
specified population hazards: calibration need not interpolate the factual
last-pre point. Inspect original-curve status and bootstrap failure reasons.

Inference and hazard pretest
----------------------------

Draw exactly B pooled samples of individuals, retaining full histories and
multiplicities, re-estimating nuisance coefficients and all effects. Use sample
SD/covariance (ddof=1), centered absolute bootstrap deviations, and inverse
empirical-CDF quantiles at sorted index ceil((1-alpha)B)-1. Pointwise bands use
coordinate pivots; simultaneous bands use the maximum over all declared post
dates. Apply uniform post weights within every draw for headline inference.
These are centered-bootstrap bands, not raw percentile or normal intervals.

P-values count centered absolute deviations at least as large as the absolute
estimate; simultaneous p-values compare the corresponding maxima. Equality is
counted conservatively. These finite-bootstrap, potentially discrete p-values
are a library companion to the paper's bands, not an additional paper theorem.

No retries, stratification, epsilon replacements or filtered-success inference
are used. Any failed effect draw suppresses the entire effect family's inference
and headline inference, retaining valid original points. Zero SE makes the
associated t/p/interval fields unavailable and disables simultaneous bands;
other valid pointwise rows and the scalar can remain available. Diagnostics
have independent bootstrap validity and cannot erase valid effect inference.

``results.pretrend_test()`` returns a defensive copy of the stored
``DurationDiDPretestResults`` Diagnostic. It never refits. Algorithm 2 contrasts
each interior pre-date's average hazard gap (CD) or cumulative hazard ratio
(PH) with the last untreated anchor, excluding baseline and anchor from the
simultaneous family. Every interior pre-date is used regardless of calibration.
Missing diagnostic moments, failed draws or invalid SEs make the entire test
unavailable, with ``reject=None``; non-rejection does not establish identification
or adequate power. Hazard contrasts are not pre-treatment outcome ATTs.

The support heuristic flags fewer than five survivors for either group at any
observed date, including factual treated post extinction. PH also flags fewer
than five control exits since baseline at each positive-weight calibration date.
Counts and warnings are reported without trimming or changing validity gates.

Results and tables
------------------

Both results own their arrays/frames and expose ``summary(alpha=None)``,
``to_dict()`` and ``to_dataframe()``. Summaries reject alpha changes; bootstrap
intervals are stored at fit time. Strict JSON serialization replaces nonfinite
numbers by null, preserves date labels as ISO strings, and serializes period-keyed
mappings as period/value records. Native tables have chronological RangeIndex.

``DurationDiDResults`` fields:

- Headline: ``att``, ``se``, ``t_stat``, ``p_value``, ``conf_int``.
- Configuration: ``method``, ``alpha``, ``seed``, ``n_bootstrap``.
- Counts: ``n_obs`` (rows), ``n_units``, ``n_treated``, ``n_control`` (individuals).
- Dates: ``periods``, ``pre_periods``, ``post_periods``, effective ``fit_periods``,
  ``requested_fit_periods``, normalized ``time_weights``, ``excluded_fit_periods``,
  ``time_origin``, ``time_step`` and method-specific ``coefficient``.
- Tables: ``effects``, ``survival_curve``, ``bootstrap_failures``.
- Bootstrap: ``bootstrap_effects`` (B by post dates), ``n_bootstrap_valid``
  (complete effect draws), optional post-ordered ``vcov``, ``cband_crit_value``.
- Availability: ``estimation_status`` (ok/invalid_counterfactual),
  ``inference_status`` and ``inference_reasons`` keyed by pointwise, simultaneous,
  simple; available/unavailable, with partial additionally allowed for pointwise.
  ``support_warnings`` and stored ``pretrend_results`` remain inspectable.
- Properties: ``inference_method='bootstrap'`` and uniform raw post mean ``raw_att``.

Native schemas, in column order:

- ``effects`` (post only): period, event_time, att, se, t_stat, p_value,
  conf_int_lower, conf_int_upper, cband_lower, cband_upper, pointwise_crit_value,
  inference_status, reason.
- ``survival_curve`` (all dates): period, elapsed_time, treated_survival,
  control_survival, treated_survivors, control_survivors,
  raw_counterfactual_cumulative_hazard, raw_counterfactual_survival, raw_att,
  counterfactual_status, reason. Pre-date extrapolations are NaN/not_estimated;
  post status is valid/invalid. Invalid original curves suppress all canonical
  effects and the headline; raw values remain here.
- ``bootstrap_failures``: draw, family, period, reason. Draws are zero-based;
  family is effects/diagnostics; period is nullable for group-level failure.
  Multiple reasons do not double-count draws. Failed effect-draw rows are all NaN.

``DurationDiDPretestResults`` holds ``method``, ``alpha``, ``anchor_period``,
``contrasts``, ``statistic``, ``p_value``, ``critical_value``, ``reject``,
``status``, ``reasons``, ``n_bootstrap``, ``n_bootstrap_attempted``,
``n_bootstrap_valid``, ``bootstrap_contrasts`` (B by tested dates), and diagnostic
``bootstrap_failures`` with the same failure schema. The contrast table has:
period, elapsed_time, contrast, se, cband_lower, cband_upper, status, reason.
It retains meaningful raw contrasts even when unavailable. With two pre-dates it
is empty with the same schema. An invalid original diagnostic attempts zero
bootstrap diagnostics and retains NaN draw slots.

``to_dataframe()`` defaults to the shared event-study schema;
``level='simple'`` returns AGGREGATION_SCHEMA, while ``'survival'`` and
``'diagnostics'`` return copies of native tables. ``aggregate('simple')`` and
``aggregate('event_study')`` are pure views, rejecting custom weights and
``balance_e``. Event-study results include reference -1 (zero effect, undefined
inference/count) and post event times starting at zero, with post covariance,
its explicit index and simultaneous bands. Counts mean treated individuals;
degrees of freedom are NaN. HonestDiD/PreTrendsPower do not admit these results.

Reporting and tutorial
----------------------

BusinessReport and DiagnosticReport expose the stored hazard diagnostic via the
native section. They do not run generic outcome parallel-trends tests. See
:doc:`business_report`, :doc:`diagnostic_report`, and
:doc:`/tutorials/33_duration_did` for availability and reporting examples.

API
---

.. autoclass:: diff_diff.DurationDiD
   :no-index:
   :members:
   :show-inheritance:

.. autoclass:: diff_diff.DurationDiDResults
   :no-index:
   :members:
   :show-inheritance:

.. autoclass:: diff_diff.DurationDiDPretestResults
   :no-index:
   :members:
   :show-inheritance:

References
----------

Deaner, B. and Ku, H. (2026), Causal Duration Analysis with Diff-in-Diff,
`arXiv:2405.05220v2 <https://arxiv.org/abs/2405.05220v2>`_. The source audit and
PH/diagnostic resolutions are in ``docs/methodology/papers/deaner-ku-2026-review.md``;
the implementation contract is in :doc:`/methodology/REGISTRY`.
