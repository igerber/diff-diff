Chained Difference-in-Differences
=================================

``ChainedDiD`` reconstructs group-time treatment effects from overlapping
within-unit outcome changes. It implements a restricted native Python version
of Bellégo, Benatia and Dortet-Bernadet's chained difference-in-differences.

.. warning::

   This is an experimental implementation. The external pinned ``cdid``
   comparison agrees on eight fixtures; an incomplete continuous-covariate
   fixture uses a different propensity-fitting criterion. Default normal
   inference still misses four small-sample coverage limits, and sparse-pair
   designs can have substantially anti-conservative intervals. See the records
   below. Algebraic
   tests and the recorded simulation scenarios are distinct evidence; they
   do not establish validity for an arbitrary incomplete panel.

Statistical contract
--------------------

Treatment is binary, absorbing and staggered, with zero anticipation and
never-treated controls. ``first_treat`` contains the positive stored treatment
date, or 0 for never-treated units. Every treatment date needs a preceding
stored date. Covariates, when supplied, must be complete and constant within
units. Treatment dates and covariates are not inferred from missing outcomes.

Every supported ordered pair of stored dates contributes a normalised IPW
comparison. The fixed identity-GMM criterion retains the full comparison
catalogue of replicated pairs, including non-adjacent dates. Identity weighting
ignores pair precision, so adding a sparse comparison can reduce precision.
It does not invert or regularise the
moment covariance and makes no optimal-efficiency claim. PT-post applies from
each cohort's reference date onwards; earlier trend differences remain free.
Reported earlier diagnostics are direct reference comparisons, with their own
support, rather than fitted GMM nuisance parameters. A non-zero earlier
difference does not refute maintained PT-post.

A sufficient baseline observation assumption is that observing both endpoints
is independent of the joint distribution of covariates and actual outcome
changes within each cohort. This includes treated changes. Without covariates,
the relevant mean restriction suffices. The estimator fits no observation model,
so treatment IPW does not correct arbitrary selective attrition. Conditional
parallel trends, no anticipation, treatment overlap and correct pair-specific
propensity models still need scientific justification.

Population and missingness
--------------------------

NaN outcomes and absent rows are unobserved endpoints; infinite outcomes and
duplicate unit-date rows are errors. The stored calendar includes dates with
only missing outcomes. Every distinct supplied unit remains in the registry,
including units with zero or one finite outcome. A unit absent from the input
is outside the declared study population. A dated row with a missing outcome
and complete static metadata can retain an otherwise unobserved unit.

``group_time_effects`` and ``to_dataframe()`` expose unavailable targets and
their support reasons. A required unsupported cell makes its overall/event-time
summary unavailable; the estimator does not silently change its population by
dropping that cell. Per-cell treated/control counts describe registry cohort
sizes; observed-pair counts and propensity diagnostics are in ``diagnostics``.

Pair eligibility and sparse support
-----------------------------------

Each pair needs at least two observed units in each arm. A singleton has no
within-arm empirical residual variance, so singleton pairs are excluded before
GMM. ``diagnostics['excluded_links']`` records every excluded candidate's counts
and reason; ``diagnostics['catalogue']`` and ``diagnostics['links']`` describe
retained pairs. Exclusions can change estimates or disconnect targets, while
all units remain in registry masses and cluster counts.

A fit warns once when singleton pairs are excluded or retained pairs have fewer
than 20 units in either arm or control effective sample size below 20. Retained
links expose ``low_support``. Two is a replication minimum and 20 is an
operational warning threshold; neither establishes calibrated inference. Two-
and five-unit arms remain particularly vulnerable. The estimator remains
experimental and statistically unqualified.

Results and inference
---------------------

``att`` averages every scheduled post-treatment cell with weight proportional
to its cohort registry size. ``aggregate('simple')`` relays that headline;
``aggregate('event_study')`` weights eligible cohorts by registry cohort size
at each actual numeric event time. Neither accepts a weighting selector or
``balance_e``. The returned :class:`~diff_diff.EventStudyResults` works with
:func:`~diff_diff.plot_event_study`.

``aggregation_weights('simple')`` and ``aggregation_weights('event_study')``
return constituent cohort/date weights and cell availability, including intended
weights for unavailable summaries. The simple container's ``weight=[1.0]``
describes its one summary row. On irregular calendars, mixed diagnostic rows
exclude structural reference identities before computing weights, counts and
covariance. Pure reference rows remain zero; ``reference_event_times`` retains
all cohort bases, whereas ``reference_periods`` identifies reference-only rows.

``vcov`` is the full estimator-scale covariance, with both axes indexed by
``vcov_index``. ``influence_functions`` stores O(1/n) original-unit
contributions. Joint uncertainty includes fitted propensity scores, estimated
cohort-share weights, shared units and shared controls. The retained
``AggregationKit`` needs no outcome panel or estimator reference.

Without ``cluster``, inference uses unit HC0 covariance. With a unit-constant
cluster column, it sums contributions by cluster and applies K/(K-1), using
all study-registry clusters. Intervals are normal, pointwise and computed at
the fit's ``alpha``. Fewer than two clusters or invalid variance yields
unavailable inference while preserving identified estimates. CR1 and normal
intervals do not establish few-cluster validity. Simultaneous bands,
prediction, survey designs, sampling weights, PT-all,
not-yet-treated controls, non-zero anticipation, outcome regression, doubly
robust estimation and conditional missingness corrections are unsupported.

Plotting preserves stored intervals when selecting an existing structural
reference explicitly. Selecting a different reference is unsupported because
chained event studies retain their fitted cohort reference dates.

Implementation and validation
-----------------------------

The methodology is specified in ``docs/methodology/chained-did-pseudocode.md``.
The V01-V16 matrix and its scope are in ``chained-did-proposal.md``. The native
qualification runner is ``benchmarks/python/validate_chained_did.py``; its
manifest records source hashes, settings, analytic targets, failures and Monte
Carlo uncertainty. ``benchmarks/R/benchmark_chained_did.R`` refuses an unmatched
R/package environment and performs no installation. Reference agreement is
not inferred from a missing or refused comparison.

.. autoclass:: diff_diff.ChainedDiD
   :no-index:
   :members:
   :inherited-members:

.. autoclass:: diff_diff.ChainedDiDResults
   :no-index:
   :members:
   :inherited-members:

Reference
---------

Bellégo, C., Benatia, D., and Dortet-Bernadet, V. (2025).
"The chained difference-in-differences." *Journal of Econometrics*.
https://doi.org/10.1016/j.jeconom.2024.105783

Methodology and validation records
----------------------------------

.. toctree::
   :maxdepth: 1

   ../methodology/chained-did-proposal
   ../methodology/chained-did-pseudocode
   ../methodology/chained-did-validation
   ../methodology/chained-did-validation-followup
   ../methodology/chained-did-review-remediation
   ../methodology/chained-did-oracle-diagnostic
