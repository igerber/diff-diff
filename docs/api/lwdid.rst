LWDiD — Lee & Wooldridge Rolling Transformation DiD
====================================================

A simple transformation approach to Difference-in-Differences estimation
that converts panel data into cross-sectional regressions (Lee & Wooldridge
2025, 2026).

The key insight from the Lee & Wooldridge papers is that, under parallel
trends and no anticipation, a unit-specific time-series transformation of
the outcome eliminates the need for two-way fixed effects entirely. For
each unit *i* with treatment onset at period *S*, Procedure 2.1 (LW 2026)
computes the pre-treatment mean:

.. math::

   \bar{Y}_{i,\text{pre}} = \frac{1}{S-1} \sum_{t=1}^{S-1} Y_{it}

and forms the transformed outcome:

.. math::

   \dot{Y}_{it} = Y_{it} - \bar{Y}_{i,\text{pre}}, \quad t = S, \ldots, T
   \qquad \text{(Equation 2.12, LW 2026)}

Under Assumption CPTC (Conditional Parallel Trends, Common Timing;
Equation 2.10, LW 2025), this transformation
removes unit-specific fixed effects, and the ATT is identified as the
coefficient on the treatment indicator in a cross-sectional regression of
:math:`\dot{Y}_{it}` on :math:`D_i` and covariates. Because the panel
problem is reduced to a cross section, *any* treatment effect estimator —
regression adjustment (RA), inverse probability weighting (IPW), doubly
robust IPWRA, or propensity-score matching — can be applied without
negative weighting, heterogeneity bias, or "bad comparisons" between
already-treated cohorts.

A second contribution (LW 2026) demonstrates that this representation
enables *exact* small-sample inference: under homoskedastic normality of
the cross-sectional error, the t-statistic follows an exact
:math:`\mathcal{T}_{N-K-2}` distribution — valid even with a single
treated unit (:math:`N_1 = 1`). When :math:`T_0` or :math:`T_1` is large,
the central limit theorem across time justifies the normality assumption
without requiring a large cross section.

.. note::

   **Why rolling transformation works.** The parallel trends assumption
   (Equation 2.15, LW 2026) implies that :math:`\Delta\bar{Y}_i(0)`
   — the difference between post-treatment and pre-treatment means of
   control potential outcomes — is mean-independent of the treatment
   indicator :math:`D_i`. This is precisely the unconfoundedness condition
   needed for cross-sectional treatment effect estimation. The
   transformation eliminates *both* unit-specific levels (via demeaning)
   and unit-specific linear trends (via detrending), weakening the
   standard parallel trends assumption to one that allows heterogeneous
   pre-intervention dynamics.

.. module:: diff_diff.lwdid

Methodology
-----------

**Procedure 2.1 — Unit-Specific Demeaning (LW 2026, Section 2)**

For common timing with intervention at period *S*:

1. Compute the pre-treatment mean for each unit:
   :math:`\bar{Y}_{i,\text{pre}} = \frac{1}{S-1}\sum_{t=1}^{S-1} Y_{it}`

2. Obtain the transformed outcome (out-of-sample residuals):

   .. math::

      \dot{Y}_{it} = Y_{it} - \bar{Y}_{i,\text{pre}}, \quad t = S, \ldots, T

3. Estimate the ATT from the cross-sectional regression (Equation 2.13, LW 2026):

   .. math::

      \dot{Y}_{it} \text{ on } 1,\; D_i, \quad i = 1, \ldots, N

The coefficient on :math:`D_i` identifies the ATT for period *t*.

**Procedure 3.1 — Unit-Specific Detrending (LW 2025, Section 5; LW 2026, Section 3)**

When parallel trends may fail but unit-specific *linear* trends capture
the pre-intervention dynamics (Assumption CHT, LW 2025):

1. For each unit *i*, regress on a constant and time over pre-treatment
   periods:

   .. math::

      Y_{it} \text{ on } 1,\; t, \quad t = 1, \ldots, S-1
      \qquad \text{(Equation 3.1, LW 2026)}

   obtaining fitted values :math:`\hat{A}_i + \hat{B}_i \cdot t`.

2. Compute the detrended outcome:

   .. math::

      \ddot{Y}_{it} = Y_{it} - \hat{A}_i - \hat{B}_i \cdot t, \quad t = S, \ldots, T
      \qquad \text{(Equation 3.2, LW 2026)}

3. Estimate the ATT from:

   .. math::

      \ddot{Y}_{it} \text{ on } 1,\; D_i, \quad i = 1, \ldots, N
      \qquad \text{(Equation 3.4, LW 2026)}

Detrending removes unit-specific intercepts :math:`\alpha_i` *and* linear
trends :math:`\beta_i t`, thus relaxing the parallel trends assumption to
allow differential pre-intervention growth rates across units (Procedure
5.1, LW 2025). This is the key advantage over Callaway & Sant'Anna (2021),
who do not accommodate heterogeneous trends.

**Procedure 4.1 — Staggered Interventions (LW 2025, Section 4)**

For staggered adoption with cohort *g* (first treatment period) and
calendar time *r*:

1. Compute the cohort-specific transformed outcome:

   .. math::

      \dot{Y}_{irg} = Y_{ir} - \frac{1}{g-1}\sum_{s=1}^{g-1} Y_{is}
      \equiv Y_{ir} - \bar{Y}_{i,\text{pre}(g)}
      \qquad \text{(Equation 4.11, LW 2025)}

2. Select the control group: units not yet treated by period *r*,
   i.e., cohorts :math:`\{r+1, \ldots, T, \infty\}`.

3. Apply any TE estimator (RA, IPW, IPWRA, matching) to the cross section
   :math:`\{(\dot{Y}_{irg}, D_{ig}, \mathbf{X}_i)\}` restricted to the
   treated cohort *g* plus control units.

Under Assumptions CNAS (conditional no anticipation, Equation 4.4) and
CPTS (conditional parallel trends, Equation 4.6), the cohort assignment
is unconfounded with respect to the transformed outcome (Theorem 4.1).

**Regression Adjustment with Interactions (Equation 3.3, LW 2025)**

When both :math:`N_0` and :math:`N_1` are sufficiently large, full
regression adjustment includes covariate interactions:

.. math::

   \dot{Y}_{ir} = \beta_0 + \beta_1 D_i + \beta_2' \mathbf{X}_i
   + \beta_3' D_i(\mathbf{X}_i - \bar{\mathbf{X}}_1) + u_i

where :math:`\bar{\mathbf{X}}_1 = N_1^{-1}\sum_{i} D_i \mathbf{X}_i` is
the mean of covariates over treated units. The ATT is :math:`\hat{\beta}_1`.
This is equivalent to separate regressions for treated and control groups
(Equation 3.3, LW 2025).

Key Assumptions
---------------

.. important::

   The LWDiD estimator requires the following assumptions for identification:

   **Assumption CPTC — Conditional Parallel Trends, Common Timing**
   (Equation 2.10, LW 2025):

   .. math::

      E[Y_{it}(0) - Y_{i1}(0) \mid D_i, \mathbf{X}_i]
      = E[Y_{it}(0) - Y_{i1}(0) \mid \mathbf{X}_i], \quad t = 2, \ldots, T

   The *trend* in control potential outcomes is independent of treatment
   assignment conditional on covariates. Note this is weaker than
   unconditional parallel trends — assignment can be correlated with
   *levels* :math:`Y_{i1}(0)`, but not with *trends*.

   **Assumption NAC — No Anticipation, Common Timing** (Equation 2.7, LW 2025):

   .. math::

      E[Y_{it}(1) - Y_{it}(0) \mid D_i = 1] = 0, \quad t = 1, \ldots, S-1

   Treatment effects are zero on average before the intervention.

   **Assumption CPTS — Conditional PT, Staggered** (Equation 4.6, LW 2025):

   .. math::

      E[Y_t(\infty) - Y_1(\infty) \mid \mathbf{D}, \mathbf{X}]
      = E[Y_t(\infty) - Y_1(\infty) \mid \mathbf{X}], \quad t = 2, \ldots, T

   Trends in the never-treated state are independent of the full vector
   of cohort assignments, enabling use of not-yet-treated units as controls.

   **Conditional Heterogeneous Trends** (Assumption CHT, Equation 5.3,
   LW 2025): When using ``detrend``, the parallel trends assumption is
   relaxed to allow unit-specific linear trends
   :math:`\eta_g \cdot t` that vary by cohort. Detrending removes these
   heterogeneous trends, restoring unconfoundedness.

Small-Sample Inference
----------------------

A distinctive feature of the LW approach (LW 2026, Section 2) is the
availability of *exact* inference. Under the classical linear model
assumptions on the cross-sectional regression:

.. math::

   U_i \mid D_i \sim \text{Normal}(0, \sigma_U^2)
   \qquad \text{(Equation 2.9, LW 2026)}

the t-statistic follows an exact Student-t distribution:

.. math::

   \frac{\hat{\tau}_{DD} - \tau}{\text{se}(\hat{\tau}_{DD})}
   \sim \mathcal{T}_{N-2}
   \qquad \text{(Equation 2.10, LW 2026)}

This holds even with :math:`N_1 = 1` (single treated unit), where the
t-statistic is interpretable as a *studentized residual* — testing whether
the treated unit is an "outlier" relative to the controls (LW 2026,
Section 2.1).

When :math:`N` is not too small, the HC3 heteroskedasticity-robust
standard error (Davidson & MacKinnon, 1993) provides reliable inference
without the homoskedasticity assumption, as shown by Simonsohn (2021).

**Randomization inference** is also supported: under the sharp null of
zero treatment effects, permutation of :math:`D_i` yields Monte Carlo
p-values without requiring normality (LW 2026, the small-sample
inference paper). Validity is conditional on the assignment mechanism the
permutation encodes — complete randomization of the treatment labels
(the treated count is held fixed); the implementation follows the
authors' package convention (inclusive Phipson-Smyth counting; see the
methodology registry's RI Note).

**HC3 caveat** — HC3 requires the leverage of every observation to be
bounded away from one; a perfectly-leveraged design (e.g. a single
treated unit) has no defined HC3 variance and fails closed with a
warning and NaN inference. Use classical exact inference there.

**PSM inference** — ``estimation_method='psm'`` reports the matched ATT
point estimate with NaN inference: no valid matching variance estimator
is currently implemented (the naive matched-pairs formula ignores
matched-control reuse and first-stage matching uncertainty; an
Abadie-Imbens variance is tracked in ``DEFERRED.md``). The contract is
enforced on every route: PSM requires ``covariates`` (there is no
propensity score without them), rejects ``n_bootstrap > 0`` in BOTH
common-timing and staggered designs (the standard bootstrap is invalid
for nearest-neighbor matching estimators — Abadie & Imbens 2008), and a
propensity-model failure falls back to a regression-adjustment POINT with
NaN inference rather than finite OLS standard errors. Use
``estimation_method='dr'`` for valid inference.

LWDiD
------

Main estimator class.

.. autoclass:: diff_diff.LWDiD
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

   .. rubric:: Methods

   .. autosummary::

      ~LWDiD.fit
      ~LWDiD.get_params
      ~LWDiD.set_params

LWDiDResults
------------

Results container returned by :meth:`~diff_diff.LWDiD.fit`.

.. autoclass:: diff_diff.lwdid_results.LWDiDResults
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Methods

   .. autosummary::

      ~LWDiDResults.summary
      ~LWDiDResults.print_summary
      ~LWDiDResults.to_dataframe
      ~LWDiDResults.to_dict

Input Contract
--------------

:meth:`~diff_diff.LWDiD.fit` validates the treatment design before any
transformation is applied. Eight requirements are enforced:

- **Absorbing treatment** — within each unit the ``treatment`` indicator
  must be non-decreasing over time: once a unit switches from 0 to 1 it
  must remain treated. Units that revert to 0 raise ``ValueError``.
- **Common timing** — when ``first_treat`` is not supplied, all treated
  units must first switch on in the same period. Heterogeneous onsets
  are rejected with a ``ValueError`` pointing to the staggered interface
  (pass ``first_treat``).
- **Staggered consistency** — when ``first_treat`` is supplied, the
  ``treatment`` indicator must satisfy :math:`D_{it} = 1[t \ge g_i]` over
  each unit's OBSERVED rows, where :math:`g_i` is the unit's
  first-treatment period; the row at :math:`t = g_i` itself may be
  unobserved (unbalanced panels with a missing onset row are accepted).
  Units that are never treated must have no treated rows.
- **Never-treated encodings** — ``first_treat`` coded ``0``, ``NaN``/
  ``NaT``, or ``np.inf`` means never-treated; ``inf`` and finite cohorts
  BEYOND the last observed period are recoded to never-treated with a
  warning (beyond-window units never switch on inside the sample).
  Negative cohorts raise. NUMERIC cohorts strictly between observed
  periods are rejected; datetime/Period cohorts map to the next observed
  period — a dtype-dependent contract documented in the methodology
  registry.
- **Variance configuration** — ``estimation_method='reg'`` accepts
  ``vcov_type`` in ``{'classical', 'hc1', 'hc2', 'hc3'}``; ``'ipw'``/
  ``'dr'``/``'psm'`` accept ``'hc1'`` only (the influence-function /
  matching variance is always used on those paths); ``cluster=`` composes
  only with ``'hc1'`` (CR1) and is rejected for ``'psm'``.
- **Never-treated units under not-yet-treated control** — when
  ``first_treat`` is supplied and ``control_group='not_yet_treated'``,
  at least one never-treated unit (``first_treat`` coded NaN or 0) must
  be present. A panel in which every unit is eventually treated raises
  ``ValueError`` rather than silently truncating the estimation sample.
- **Unit-constant covariates** — ``covariates`` (and a non-unit
  ``cluster=`` column) must be constant within each unit on BOTH timing
  paths; time-varying columns raise ``ValueError`` (LWDiD collapses the
  panel to one row per unit, so a time-varying value would make the
  estimate depend on row order).
- **Distinct, non-reserved column names** — the core role columns
  (outcome/unit/time/treatment/``first_treat``) must be pairwise
  distinct, covariates may not repeat a core role, and no role column
  may use an LWDiD-internal working name (``_treat``, ``_ydot``,
  ``_ydot_avg``, ``_ever_treated``, ``_boot_unit``, ``_lwdid_time_pos``,
  ``_lwdid_cohort_pos``, ``_lwdid_season``) — a collision would silently overwrite the
  internal column (e.g. ``cluster='_treat'`` previously reported the
  cluster labels' coefficient as the ATT). ``cluster=`` equal to the
  unit column remains supported.

.. note::

   **Bootstrap scope and reproducibility.** In common-timing fits
   ``n_bootstrap`` activates a unit-resampling (or cluster-resampling,
   under ``cluster=``) bootstrap for the overall ATT; the headline
   se/p-value/CI then come from the bootstrap while ``params``/``vcov``
   remain the analytical regression quantities, recorded via
   ``inference_basis`` (``'unit_bootstrap'``/``'cluster_bootstrap'``) and
   rendered by ``summary()``. The per-replicate RNG streams are
   ``SeedSequence``-spawned identically for every ``n_jobs``, so a seeded
   fit reproduces exactly across serial and parallel execution. In STAGGERED fits
   ``n_bootstrap`` governs the event-study multiplier bootstrap only
   (sup-t simultaneous bands); the overall and cohort aggregates keep
   analytical influence-function inference, with the per-surface basis
   recorded on the results object (``cband_method``,
   ``cband_n_bootstrap``, ``inference_basis``). Event cells whose
   multiplier draws are degenerate fail closed (point retained, NaN
   inference) rather than silently reverting to analytical standard
   errors. The common-timing event-study surface covers post periods
   only; pre-treatment placebo cells are produced by the staggered path
   (pass ``first_treat=``, which for a single cohort matches the
   common-timing regression on single-post-period panels).

.. note::

   :meth:`~diff_diff.lwdid_results.LWDiDResults.to_dict` returns only
   JSON-native types: numpy scalars and arrays are converted to Python
   ints/floats/bools and lists, and datetime-like labels (Timestamp,
   Period) become strings, so ``json.dumps(result.to_dict())`` works
   directly.

Example Usage
-------------

**Basic demeaning with regression adjustment (Procedure 2.1):**

.. code-block:: python

   import pandas as pd
   from diff_diff import LWDiD, generate_staggered_data

   # Generate staggered panel data; the 'treated' column is the binary
   # indicator D_it = 1[period >= first_treat] (0 for never-treated units)
   data = generate_staggered_data(n_units=200, n_periods=10,
                                  cohort_periods=[4, 7], seed=42)

   # Procedure 2.1: demean + reg estimates the ATT via cross-sectional OLS
   # on the transformed outcome Y_dot = Y_post - Y_bar_pre
   lw = LWDiD(rolling="demean", estimation_method="reg", vcov_type="hc1")
   results = lw.fit(data, outcome="outcome", unit="unit",
                    time="period", treatment="treated",
                    first_treat="first_treat")
   results.print_summary()

**Doubly-robust IPWRA estimation (Procedure 3.1, Step 2):**

.. code-block:: python

   # DR (IPWRA) combines propensity score weighting with regression adjustment
   # on the transformed outcome — doubly robust as in Wooldridge (2007).
   # Cluster-robust inference activates via the constructor's cluster= parameter.
   data["state"] = data["unit"] % 40  # cluster identifier
   lw_dr = LWDiD(rolling="demean", estimation_method="dr", cluster="state")
   results_dr = lw_dr.fit(data, outcome="outcome", unit="unit",
                          time="period", treatment="treated",
                          first_treat="first_treat")
   print(f"ATT: {results_dr.att:.4f} (SE={results_dr.se:.4f})")

**Staggered adoption with detrending (Procedure 4.1 + 5.1):**

.. code-block:: python

   # Detrending removes unit-specific linear trends before estimation,
   # relaxing parallel trends to allow heterogeneous pre-intervention dynamics
   lw_stag = LWDiD(rolling="detrend", control_group="never_treated")
   results_stag = lw_stag.fit(data, outcome="outcome", unit="unit",
                              time="period", treatment="treated",
                              first_treat="first_treat")
   # Cohort-specific ATT(g) estimates (Equations 7.2/7.10, LW 2026).
   # Aggregation convention: each cohort's estimable cells weight by
   # their contributing treated mass (cell-mass, matching the WATT(r)
   # axis); this equals the eq. 7.10 unit-average estimand on balanced
   # never-treated designs and deviates on unbalanced panels (see the
   # methodology registry's within-cohort aggregation Note).
   df_cohorts = results_stag.to_dataframe()
   print(df_cohorts)

**Robustness check — demean vs detrend (informal pre-test for trend
sensitivity):**

.. code-block:: python

   # Comparing demean vs detrend provides a specification robustness check.
   # If results differ substantially, it suggests unit-specific trends matter
   # (see LW 2025, Section 6 — Walmart application, Figure 1 panels b vs c)
   for transform in ("demean", "detrend"):
       lw_check = LWDiD(rolling=transform, estimation_method="dr", vcov_type="hc1")
       res = lw_check.fit(data, outcome="outcome", unit="unit",
                          time="period", treatment="treated",
                          first_treat="first_treat")
       print(f"{transform}: ATT={res.att:.4f} (SE={res.se:.4f})")

Wild cluster bootstrap
----------------------

``diff_diff.lwdid_wild_bootstrap.wild_cluster_bootstrap(y, treatment,
cluster_ids, controls=None, *, n_bootstrap=999, weight_type='rademacher',
alpha=0.05, seed=None)`` provides few-cluster inference on a collapsed
cross-section. It delegates to the house Wild Cluster Restricted engine
(:func:`diff_diff.wild_bootstrap_se`, matched to R's
``fwildclusterboot::boottest``): the null is imposed by dropping the
treatment column while keeping the controls, the confidence interval is
obtained by test inversion, and Rademacher weights are fully enumerated
automatically when :math:`2^G \le` ``n_bootstrap``. The result carries
``att``, the analytical CR1 ``se``, ``t_stat_original``, ``p_value``
(strict-exceedance house convention), the test-inversion
``ci_lower``/``ci_upper``, ``n_clusters``, ``n_bootstrap``,
``weight_type``, ``alpha``, the finite-filtered ``bootstrap_distribution``
(``None`` when the degenerate-design guard fires), and ``n_dropped``
(non-finite outcome rows dropped with a warning). Exactly-identified
designs (cluster-invariant treatment with zero cluster scores) fail
closed: the point estimate is retained with NaN inference.

The RESULT-LEVEL methods ``LWDiDResults.wild_cluster_bootstrap()`` and
``LWDiDResults.randomization_test()`` take no data arguments: they REPLAY
the fit-time collapsed cross-section and the exact fitted RA design
(including the treatment-centered covariate interactions; randomization
draws recompute the treated covariate mean per assignment), and assert
the replayed statistic equals ``.att`` before caching a p-value — so
``bootstrap_pvalue``/``ri_pvalue`` always describe the fitted estimand.
They are defined for common-timing ``estimation_method='reg'`` fits
(``wild_cluster_bootstrap`` additionally requires a ``cluster=`` fit);
use the standalone module functions above for generic arrays.


Empirical Applications
----------------------

The Lee & Wooldridge papers validate the methodology with two empirical
studies:

- **California Proposition 99** (LW 2026, Section 6): With a single treated
  state (:math:`N_1 = 1`) and 38 control states, Procedure 3.1
  (unit-specific detrending) achieves an excellent pre-treatment fit and
  yields a per-period treatment trajectory that grows over time — from
  :math:`\hat{\tau}_{1989} = -0.043` (SE = 0.059) to
  :math:`\hat{\tau}_{2000} = -0.403` (SE = 0.152). The exact-inference
  p-value (0.021) is valid under the conditional-normality and
  homoskedasticity assumptions — it tests the treatment-effect null, not
  those assumptions themselves; randomization inference (below) is a
  robustness check that does not require normality, conditional on the
  complete-randomization assignment mechanism. (The paper's printed
  randomization-inference p-value of 0.020 is not reproducible with the
  authors' own package, which implements the inclusive Phipson-Smyth rule
  and converges to ~0.051 at 100k replications — see the methodology
  registry's RI Note; the implementation follows the package convention.) This demonstrates the
  method works with as few as one treated unit.

- **Walmart minimum-wage study** (LW 2025, Section 6): A balanced panel of
  1,280 counties over 23 years, with staggered Walmart openings. The
  rolling IPWRA estimator with detrending (Procedure 5.1) reveals that
  county-level linear trends are critical: the CS (2021) estimate of 5.4%
  employment increase shrinks to 3.2% (SE = 0.5%) once heterogeneous
  trends are removed — the latter consistent with Basker's (2005) estimate
  of 150–300 new retail jobs per Walmart store.

- **Castle doctrine laws** (LW 2026, Section 7.2): A staggered rollout
  across 21 states (2005–2009), with 29 never-treated controls. The
  aggregated ATT :math:`\hat{\tau}_\omega = 0.092` (9.2% increase in
  homicides) is obtained from a single cross-sectional regression
  (Equation 7.19, LW 2026), with the HC3 t-statistic of 1.50.

Estimator Comparison
--------------------

.. list-table:: LWDiD vs. CallawaySantAnna vs. WooldridgeDiD
   :header-rows: 1
   :widths: 20 27 27 26

   * - Feature
     - LWDiD
     - CallawaySantAnna
     - WooldridgeDiD
   * - Approach
     - Unit-specific transform → cross-sectional TE estimation
     - Long-difference :math:`Y_{it} - Y_{i,g-1}` (Eq. 4.13, LW 2025)
     - Single saturated POLS/TWFE regression
   * - Pre-treatment info
     - All periods :math:`\{1,\ldots,g-1\}` (rolling average)
     - Only period :math:`g-1` (long difference)
     - All periods (full regression)
   * - Key identification
     - Unconfoundedness of :math:`D_i` w.r.t. :math:`\dot{Y}(0)` (Thm 4.1)
     - PT on first differences
     - Mundlak-style cohort×time interactions
   * - Estimators
     - RA, IPW, IPWRA, PSM, matching
     - OR, IPW, DR
     - OLS, Poisson, Logit
   * - Heterogeneous trends
     - Yes (detrend, Procedure 5.1)
     - No
     - No
   * - Exact small-N inference
     - Yes (:math:`\mathcal{T}_{N-2}` under CLM, Eq. 2.10 LW 2026)
     - No (requires large N)
     - No (requires large N)
   * - Doubly robust
     - Yes (IPWRA)
     - Yes (DR)
     - No (single equation)
   * - Efficiency (common timing)
     - BLUE + asymptotically efficient (Theorem 3.1, LW 2025)
     - Less efficient (uses only :math:`g-1`)
     - Equivalent to LW RA (Theorem 3.1)

Restrictions
------------

.. warning::

   The following restrictions apply to the current implementation:

- **At least 2 pre-treatment observations per unit for detrend** — the ``detrend`` transformation
  fits a unit-specific linear trend on pre-treatment observations; units
  with fewer than 2 pre-treatment periods cannot be detrended and are
  dropped with a ``UserWarning``.
- **Binary absorbing treatment** — the ``treatment`` column must be a binary
  indicator that switches from 0 to 1 and stays on. Non-binary or
  non-absorbing treatment raises ``ValueError``.
- **PSM matching** — when ``estimation_method='psm'``, unmatched treated units
  (no control within ``caliper``) receive NaN and are excluded from the
  ATT. A ``UserWarning`` reports the count of dropped treated units.
- **Propensity score trimming** — IPW/DR clip estimated propensity scores
  to ``[pscore_trim, 1 - pscore_trim]`` (default 0.01/0.99) for
  numerical stability. Extreme scores indicate poor overlap (violation of
  Assumption OVLS, Equation 4.10, LW 2025).
- **Per-period effects** — per-period (event-study) effects live on the
  unified post-fit surface: call ``results.aggregate('event_study')`` on
  the fitted :class:`~diff_diff.lwdid_results.LWDiDResults`.
- **Not-yet-treated control** — when ``control_group='not_yet_treated'``,
  the set of valid controls for cohort *g* at time *r* comprises units
  with :math:`D_{i,r+1} + \cdots + D_{iT} + D_{i\infty} = 1`
  (Equation 4.12, LW 2025). This excludes already-treated cohorts,
  preventing "bad comparisons."

.. seealso::

   :class:`~diff_diff.CallawaySantAnna`
      Propensity-score reweighting using long differences (Equation 4.13, LW 2025).
   :class:`~diff_diff.WooldridgeDiD`
      Mundlak-style saturated regression — equivalent to RA under LWDiD for
      common timing (Theorem 3.1, LW 2025).
   :class:`~diff_diff.ImputationDiD`
      FE imputation approach (Borusyak, Jaravel & Spiess 2024).
