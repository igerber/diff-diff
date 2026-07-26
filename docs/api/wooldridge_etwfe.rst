Wooldridge Extended Two-Way Fixed Effects (ETWFE)
===================================================

Extended Two-Way Fixed Effects estimator from Wooldridge (2025, 2023),
based on the Stata ``jwdid`` package specification (Friosavila 2021),
with documented SE/aggregation deviations noted in the Methodology Registry.

This module implements ETWFE via a single saturated regression that:

1. **Estimates ATT(g,t)** for each cohort×time treatment cell simultaneously
2. **Supports linear (OLS), Poisson QMLE, and logit** link functions
3. **Uses ASF-based ATT** for nonlinear models: E[f(η₁)] − E[f(η₀)]
4. **Computes delta-method SEs** for all aggregations (event, group, calendar, simple)
5. **Supports paper W2025 cohort-share aggregation** via ``aggregate(weights="cohort_share")`` (Eqs. 7.4 + 7.6; default is cell-count matching Stata ``jwdid_estat``)
6. **Supports paper W2025 Section 8 heterogeneous cohort trends** via ``cohort_trends=True`` (OLS path only; auto-routes to full-dummy mode; requires ``control_group="not_yet_treated"`` — the default — and ``survey_design=None``; the ``never_treated`` and survey paths are fail-closed with ``NotImplementedError`` because the placebo-cell basis remaining collinear with the trend columns through the unit fixed effects / unvalidated survey-TSL composition would make the trend specification unidentified or unverified — see Methodology Registry for the full contract)
7. **Follows the Stata jwdid specification** for OLS defaults and nonlinear paths (see Methodology Registry for documented SE/aggregation deviations)

**When to use WooldridgeDiD:**

- Staggered adoption design with heterogeneous treatment timing
- Nonlinear outcomes (binary, count, non-negative continuous)
- You want a single-regression approach matching Stata's ``jwdid``
- You need event-study, group, calendar, or simple ATT aggregations
- You need paper W2025 cohort-share aggregation weights as an alternative
  to the default cell-count weighting
- You need heterogeneous cohort-specific linear trends when parallel
  trends is violated (paper W2025 Section 8)

**References:**

- Wooldridge, J. M. (2025). Two-way fixed effects, the two-way Mundlak
  regression, and difference-in-differences estimators. *Empirical
  Economics*, 69(5), 2545-2587. DOI 10.1007/s00181-025-02807-z.
- Wooldridge, J. M. (2023). Simple approaches to nonlinear
  difference-in-differences with panel data. *The Econometrics Journal*,
  26(3), C31-C66.
- Friosavila, F. (2021). ``jwdid``: Stata module for ETWFE. SSC s459114.

.. module:: diff_diff.wooldridge

WooldridgeDiD
--------------

Main estimator class for Wooldridge ETWFE.

.. autoclass:: diff_diff.WooldridgeDiD
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Methods

   .. autosummary::

      ~WooldridgeDiD.fit
      ~WooldridgeDiD.get_params
      ~WooldridgeDiD.set_params

WooldridgeDiDResults
---------------------

Results container returned by ``WooldridgeDiD.fit()``.

``cohort_trend_coefs`` (populated under ``cohort_trends=True``, OLS path
only): ``Dict[g → δ_g]`` keyed by treated cohort. The reported slopes
are **relative to the baseline trend** absorbed by the design — the
never-treated cohort's trend (when a never-treated cohort exists) OR
the last cohort's trend (when no never-treated cohort exists, per
paper W2025 Section 5.4's all-eventually-treated drop rule). On
all-treated panels the last cohort is intentionally absent from the
dict; its slope is the baseline (zero in deviation form).

.. warning::

   The all-eventually-treated branch is currently **unreachable**: the
   paper's Section 5.4 rule is applied to the trend columns but NOT to
   the cohort × time cells, so such panels are rank-deficient at
   fully-treated periods and ``fit()`` raises rather than returning
   relabeled contrasts. Implementing the cell half restores these fits
   (tracked in ``TODO.md``).

See ``docs/methodology/REGISTRY.md`` → ``## WooldridgeDiD (ETWFE)`` →
"Heterogeneous cohort trends" for the full normalization contract.

.. autoclass:: diff_diff.wooldridge_results.WooldridgeDiDResults
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Methods

   .. autosummary::

      ~WooldridgeDiDResults.aggregate
      ~WooldridgeDiDResults.summary

Example Usage
-------------

Basic OLS (follows Stata ``jwdid y, ivar(unit) tvar(time) gvar(cohort)``)::

    import pandas as pd
    from diff_diff import WooldridgeDiD

    df = pd.read_stata("mpdta.dta")
    df['first_treat'] = df['first_treat'].astype(int)

    m = WooldridgeDiD()
    r = m.fit(df, outcome='lemp', unit='countyreal', time='year', cohort='first_treat')

    r.aggregate('event').aggregate('group').aggregate('simple')
    print(r.summary('event'))
    print(r.summary('group'))
    print(r.summary('simple'))

.. note::

   When ``method="ols"`` is applied to a binary (``{0, 1}``) or non-negative
   integer-count outcome, ``fit()`` emits a ``UserWarning`` noting that a
   matching nonlinear model (``method="logit"`` / ``method="poisson"``) is often
   the *more appropriate* specification for such outcomes — it imposes parallel
   trends on the link/index scale rather than in levels (Wooldridge 2023 notes
   level-PT is only valid for continuous/unbounded outcomes), and in that
   paper's simulations the linear model is both biased and less precise where
   the nonlinear mean holds. It rests on a *different identifying assumption*
   than linear OLS, so treat it as a recommended comparison, not an automatic
   switch. OLS remains a valid QMLE for *any* response (Wooldridge 2023);
   suppress the hint via ``warnings.filterwarnings``. The check is heuristic:
   bounded discrete (binomial-style) outcomes with a known upper bound are not
   separately detected from unbounded counts.

View cohort×time cell estimates (post-treatment)::

    for (g, t), v in sorted(r.group_time_effects.items()):
        if t >= g:
            print(f"g={g} t={t}  ATT={v['att']:.4f}  SE={v['se']:.4f}")

Poisson QMLE for non-negative outcomes
(follows Stata ``jwdid emp, method(poisson)``)::

    import numpy as np
    df['emp'] = np.exp(df['lemp'])

    m_pois = WooldridgeDiD(method='poisson')
    r_pois = m_pois.fit(df, outcome='emp', unit='countyreal',
                        time='year', cohort='first_treat')
    r_pois.aggregate('event').aggregate('group').aggregate('simple')
    print(r_pois.summary('simple'))

Logit for binary outcomes
(follows Stata ``jwdid y, method(logit)``)::

    m_logit = WooldridgeDiD(method='logit')
    r_logit = m_logit.fit(df, outcome='hi_emp', unit='countyreal',
                          time='year', cohort='first_treat')
    r_logit.aggregate('group').aggregate('simple')
    print(r_logit.summary('group'))

Aggregation Methods
-------------------

Call ``.aggregate(type, weights=...)`` before ``.summary(type)``:

.. list-table::
   :header-rows: 1
   :widths: 15 30 25

   * - Type
     - Description
     - Stata equivalent
   * - ``'event'``
     - ATT by relative time k = t − g
     - ``estat event``
   * - ``'group'``
     - ATT averaged across post-treatment periods per cohort
     - ``estat group``
   * - ``'calendar'``
     - ATT averaged across cohorts per calendar period
     - ``estat calendar``
   * - ``'simple'``
     - Overall weighted average ATT
     - ``estat simple``

**Weighting schemes** (``weights="cell"`` default, ``weights="cohort_share"``
opt-in):

- ``weights="cell"`` (default) — cell-count ``n_{g,t}`` weighting; matches
  Stata ``jwdid_estat``. Supported for all four aggregation types.
- ``weights="cohort_share"`` — paper W2025 Eq. 7.4 (simple) and Eq. 7.6
  (event, restricted to ``k >= 0``) cohort-share weighting. Supported
  only for ``type="simple"`` and ``type="event"``; raises on
  ``type ∈ {"group","calendar"}`` (no paper closed-form). Inference
  fields (t-stat / p-value / conf-int) are fail-closed to ``NaN``
  with a ``UserWarning`` documenting the conditional-on-shares
  limitation (paper W2025 Section 7.5). Raises on
  ``survey_design is not None`` (design-consistent cohort totals
  pending follow-up).

Comparison with Other Staggered Estimators
------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 27 27 26

   * - Feature
     - WooldridgeDiD (ETWFE)
     - CallawaySantAnna
     - ImputationDiD
   * - Approach
     - Single saturated regression
     - Separate 2×2 DiD per cell
     - Impute Y(0) via FE model
   * - Nonlinear outcomes
     - Yes (Poisson, Logit)
     - No
     - No
   * - Covariates
     - Via regression (linear index)
     - OR, IPW, DR
     - Supported
   * - SE for aggregations
     - Delta method
     - Multiplier bootstrap
     - Multiplier bootstrap
   * - Stata equivalent
     - ``jwdid``
     - ``csdid``
     - ``did_imputation``
