.. meta::
   :description: Guide to choosing the right Difference-in-Differences estimator. Covers basic DiD, TWFE, staggered adoption methods (Callaway-Sant'Anna, Sun-Abraham), Synthetic DiD, and more.
   :keywords: which DiD estimator, staggered DiD estimator, difference-in-differences method selection, TWFE alternatives

Choosing an Estimator
=====================

This guide helps you select the right estimator for your research design.

Decision Flowchart
------------------

Start here and follow the questions:

0. **Is this a triple-difference (DDD) design?** (Two criteria for treatment: e.g., policy adoption AND group eligibility)

   - **No** → Go to question 1
   - **Yes, simultaneous treatment (2×2×2)** → Use :class:`~diff_diff.TripleDifference`
   - **Yes, with staggered timing** → Use :class:`~diff_diff.TripleDifference` with ``first_treat=`` (:class:`~diff_diff.StaggeredTripleDifference` is deprecated in 3.9 and runs the same engine until its 4.0 removal)

1. **Is treatment continuous?** (Units receive different doses or intensities)

   - **No** → Go to question 2
   - **Yes** → Use :class:`~diff_diff.ContinuousDiD`

2. **Can treatment switch on AND off?** (Reversible / non-absorbing treatment — e.g., marketing campaigns, seasonal promotions, on/off policy cycles)

   - **No (treatment is absorbing — once treated, stays treated)** → Go to question 3
   - **Yes** → Use :class:`~diff_diff.ChaisemartinDHaultfoeuille` — the most general option (allows dynamic/carryover effects, with joiner/leaver views). :class:`~diff_diff.LPDiD` (``non_absorbing="first_entry"`` / ``"effect_stabilization"``) and :class:`~diff_diff.TROP` (``non_absorbing=True``, under a no-dynamic-effects assumption) also handle non-absorbing treatment under stronger assumptions

3. **Is treatment staggered?** (Different units treated at different times)

   - **No** → Go to question 4
   - **Yes** → Use :class:`~diff_diff.CallawaySantAnna` (or :class:`~diff_diff.EfficientDiD` for tighter SEs under PT-All)
   - **Yes, and you suspect homogeneous effects** → Use :class:`~diff_diff.ImputationDiD` or :class:`~diff_diff.TwoStageDiD` for tighter CIs
   - **Yes, with nonlinear outcome (binary/count)** → Use :class:`~diff_diff.WooldridgeDiD` with ``method='logit'`` or ``method='poisson'``
   - **Want to diagnose TWFE bias?** → Use :class:`~diff_diff.BaconDecomposition` first

4. **Do you have panel data?** (Multiple observations per unit over time)

   - **No** → Use :class:`~diff_diff.DifferenceInDifferences` (basic 2x2)
   - **No, and you care about effect heterogeneity across the outcome distribution** → Use :class:`~diff_diff.ChangesInChanges` (2x2 quantile treatment effects, invariant to monotone outcome rescaling in unconditional fits; optional numeric covariates via quantile-regression conditioning - the covariate branch's linear quantile regressions are not equivariant to nonlinear monotone transforms; works with panel data too - ``panel=True`` changes only the bootstrap). :class:`~diff_diff.ChangesInChanges` with ``method="qdid"`` is the quantile-DiD comparison estimator (the standalone ``QDiD`` class is deprecated in 3.9); Athey & Imbens (2006) recommend CiC over it
   - **Yes** → Go to question 5

5. **Do you need period-specific effects?** (Event study design)

   - **No** → Use :class:`~diff_diff.TwoWayFixedEffects`
   - **Yes** → Use :class:`~diff_diff.TwoWayFixedEffects` with
     ``event_study=True`` (``MultiPeriodDiD`` is deprecated in 3.9;
     ``spec="pooled"`` reproduces its design)

6. **Is your treated group small?** (Few treated units, many controls)

   - Consider :class:`~diff_diff.SyntheticDiD` for better pre-treatment fit

Quick Reference
---------------

.. list-table::
   :header-rows: 1
   :widths: 20 30 25 25

   * - Estimator
     - Best For
     - Key Assumption
     - Output
   * - ``DifferenceInDifferences``
     - Simple 2x2 designs, cross-sectional comparisons
     - Parallel trends (2 periods)
     - Single ATT
   * - ``TwoWayFixedEffects``
     - Panel data, simultaneous treatment
     - Parallel trends (all periods)
     - Single ATT with unit/time FE
   * - ``MultiPeriodDiD`` (deprecated 3.9 → ``TwoWayFixedEffects`` ``event_study=True``)
     - Event studies, dynamic effects
     - Parallel trends (pre-periods)
     - Period-specific effects
   * - ``CallawaySantAnna``
     - Staggered adoption, heterogeneous timing
     - Conditional parallel trends
     - Group-time ATT(g,t), aggregations
   * - ``ChaisemartinDHaultfoeuille``
     - Reversible / non-absorbing treatments (most general; allows dynamic effects)
     - Parallel trends + A5 (no crossing) + A11 (stable controls)
     - DID_l event study (L_max), normalized DID^n_l, cost-benefit delta, placebos, sup-t bands, TWFE diagnostic
   * - ``SyntheticDiD``
     - Few treated units, many controls
     - Synthetic parallel trends
     - ATT with unit/time weights
   * - ``EfficientDiD``
     - Staggered adoption with optimal efficiency
     - PT-All (overidentified) or PT-Post
     - Group-time ATT(g,t), aggregations
   * - ``ContinuousDiD``
     - Continuous dose / treatment intensity
     - Strong Parallel Trends (SPT) for dose-response; PT for binarized ATT
     - ATT\ :sup:`loc` (PT); ATT(d), ACRT(d) (SPT)
   * - ``HeterogeneousAdoptionDiD``
     - Universal rollout, dose varies, no untreated unit
     - dCDH 2026 Assumptions (Design 1' QUG case or Design 1 with A6/A5)
     - WAS or WAS\ :sub:`d_lower` per resolved estimand; event-study Appendix B.2
   * - ``SunAbraham``
     - Staggered adoption, interaction-weighted
     - Conditional parallel trends
     - Cohort-specific ATTs, event study
   * - ``ImputationDiD``
     - Staggered, homogeneous effects
     - Unit + time FE structure
     - Imputed treatment effects, event study
   * - ``TwoStageDiD``
     - Staggered adoption, efficient
     - Unit + time FE structure
     - Single ATT or event study
   * - ``StackedDiD``
     - Staggered, sub-experiment approach
     - Parallel trends per cohort
     - Trimmed aggregate ATT
   * - ``TROP``
     - Factor confounding suspected
     - Factor model + weights
     - ATT with triple robustness
   * - ``TripleDifference``
     - Two eligibility criteria (DDD)
     - Parallel trends for both dimensions
     - DDD ATT (regression, IPW, or DR)
   * - ``StaggeredTripleDifference``
     - Staggered DDD with treatment timing
     - Conditional parallel trends (DDD)
     - Group-time ATT(g,t), aggregations
   * - ``WooldridgeDiD``
     - Nonlinear outcomes or saturated OLS
     - Conditional parallel trends
     - OLS: direct coefficients; logit/Poisson: ASF-based ATT
   * - ``LPDiD``
     - Fast staggered (absorbing) event studies without negative weighting
     - Parallel trends, no anticipation; absorbing treatment
     - Event-study path + pooled pre/post ATT
   * - ``ChangesInChanges``
     - 2x2 distributional effects (which quantiles moved, not just the mean)
     - h(u, t) monotone in a scalar unobservable; U time-invariant within groups
     - ATT + quantile treatment effects (bootstrap inference)
   * - ``QDiD`` (deprecated 3.9; use ``ChangesInChanges(method="qdid")``)
     - 2x2 quantile-DiD comparison alongside ChangesInChanges
     - Additive quantile model (scale-dependent, testable restrictions)
     - ATT + quantile treatment effects (bootstrap inference)
   * - ``BaconDecomposition``
     - TWFE diagnostic
     - (diagnostic tool)
     - 2x2 decomposition weights

Detailed Guidance
-----------------

Basic 2x2 DiD
~~~~~~~~~~~~~

Use :class:`~diff_diff.DifferenceInDifferences` when:

- You have a simple before/after, treatment/control design
- Treatment occurs simultaneously for all treated units
- You want a single average treatment effect

.. code-block:: python

   from diff_diff import DifferenceInDifferences

   did = DifferenceInDifferences()
   results = did.fit(data, outcome='y', treatment='treated', post='post')

Two-Way Fixed Effects
~~~~~~~~~~~~~~~~~~~~~

Use :class:`~diff_diff.TwoWayFixedEffects` when:

- You have panel data with multiple time periods
- Treatment timing is the same for all treated units
- You want to control for unit and time fixed effects
- You don't need to see period-by-period effects

.. warning::

   TWFE can be biased with staggered treatment timing. Already-treated units
   act as controls for newly-treated units, which can cause negative weighting.
   Use :class:`~diff_diff.CallawaySantAnna` for staggered designs.

.. code-block:: python

   from diff_diff import TwoWayFixedEffects

   twfe = TwoWayFixedEffects()
   results = twfe.fit(data, outcome='y', treatment='treated',
                      unit='unit_id', post='post')

Multi-Period Event Study
~~~~~~~~~~~~~~~~~~~~~~~~

Use :class:`~diff_diff.TwoWayFixedEffects` with ``event_study=True`` when:

- You want a full event-study with pre and post treatment effects
- You need pre-period coefficients to assess parallel trends
- You want to visualize treatment effect dynamics over time
- All treated units receive treatment at the same time (simultaneous adoption)

The default ``spec="within"`` estimates the unit-FE event study;
``spec="pooled"`` reproduces the design of the deprecated
:class:`~diff_diff.MultiPeriodDiD` (removed in 4.0) and is the only spec
valid for repeated cross-sections.

.. code-block:: python

   from diff_diff import TwoWayFixedEffects, plot_event_study

   event = TwoWayFixedEffects()
   results = event.fit(data, outcome='y', treatment='treated',
                       unit='unit_id', event_study=True,
                       time='period', post_periods=[3, 4, 5],
                       reference_period=2)

   # Visualize
   plot_event_study(results)

Callaway-Sant'Anna
~~~~~~~~~~~~~~~~~~

Use :class:`~diff_diff.CallawaySantAnna` when:

- Treatment is adopted at different times (staggered rollout)
- You want valid treatment effect estimates with heterogeneous timing
- You need group-time specific effects ATT(g,t)

This is the recommended estimator for most applied work with staggered adoption.

.. code-block:: python

   from diff_diff import CallawaySantAnna

   cs = CallawaySantAnna(
       control_group='never_treated',  # or 'not_yet_treated'
       estimation_method='dr'  # doubly robust (recommended)
   )
   results = cs.fit(data, outcome='y', unit='unit_id',
                    time='period', first_treat='first_treat',
                    covariates=['x1', 'x2'])

   # Overall ATT
   print(f"Overall ATT: {results.overall_att:.3f}")

   # Event study aggregation (post-fit - no refit needed)
   es = results.aggregate('event_study')
   event_study_df = es.to_dataframe()

Reversible (Non-Absorbing) Treatment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use :class:`~diff_diff.ChaisemartinDHaultfoeuille` (alias :class:`~diff_diff.DCDH`) when:

- Treatment can switch on **and** off over time (e.g., marketing campaigns,
  seasonal promotions, on/off policy cycles)
- You need separate joiners (``DID_+``) and leavers (``DID_-``) views, plus
  the aggregate ``DID_M``
- You want a built-in placebo and a TWFE decomposition diagnostic computed
  on the data you pass in (pre-filter) for direct comparison against
  ``DID_M``
- You want a multi-horizon event study (pass ``L_max`` to ``fit()``) with
  normalized effects, cost-benefit aggregation, dynamic placebos, and
  sup-t simultaneous confidence bands

This is the **most general** library estimator for non-absorbing treatment: it
allows dynamic (carryover) effects and reports separate joiner/leaver views.
Two other estimators also accept non-absorbing treatment under stronger
assumptions: :class:`~diff_diff.LPDiD` (``non_absorbing="first_entry"`` /
``"effect_stabilization"`` — entry-effect estimands) and :class:`~diff_diff.TROP`
(``non_absorbing=True``, ``method='local'`` — valid under the paper's
no-dynamic-effects / no-carryover assumption). The remaining staggered estimators
(:class:`~diff_diff.CallawaySantAnna`, :class:`~diff_diff.SunAbraham`,
:class:`~diff_diff.ImputationDiD`, :class:`~diff_diff.TwoStageDiD`,
:class:`~diff_diff.EfficientDiD`, :class:`~diff_diff.WooldridgeDiD`) assume
treatment is absorbing - once treated, stays treated.

Ships ``DID_M`` (= ``DID_1``) from de Chaisemartin & D'Haultfœuille
(2020), the full multi-horizon event study ``DID_l`` for ``l = 1..L_max``
from the dynamic companion paper (NBER WP 29873), residualization-style
covariate adjustment (``controls``), group-specific linear trends
(``trends_linear``), state-set-specific trends (``trends_nonparam``),
heterogeneity testing, non-binary treatment, HonestDiD sensitivity
integration on placebos, and survey support via Taylor-series linearization.

.. code-block:: python

   from diff_diff import ChaisemartinDHaultfoeuille
   from diff_diff.prep import generate_reversible_did_data

   data = generate_reversible_did_data(n_groups=80, n_periods=6, seed=42)

   est = ChaisemartinDHaultfoeuille()
   results = est.fit(
       data,
       outcome="outcome",
       unit="group",
       time="period",
       treatment="treatment",
   )
   results.print_summary()

   print(f"DID_M (overall): {results.overall_att:.3f}")
   print(f"DID_+ (joiners): {results.joiners_att:.3f}")
   print(f"DID_- (leavers): {results.leavers_att:.3f}")
   print(f"Placebo:         {results.placebo_effect:.3f}")

.. note::

   By default, the estimator drops groups whose treatment switches more
   than once before estimation (``drop_larger_lower=True``, matching the R
   ``DIDmultiplegtDYN`` reference). This is required for the analytical
   variance formula to be consistent with the point estimate. Each drop
   emits an explicit warning.

.. note::

   Single-period placebo ``DID_M^pl`` (``L_max=None``) has ``NaN`` SE -
   the per-period aggregation path has no influence-function derivation,
   so inference fields stay ``NaN`` even when ``n_bootstrap > 0``. The
   point estimate is meaningful for visual pre-trends inspection.
   Multi-horizon dynamic placebos ``DID^{pl}_l`` (``L_max >= 1``) have
   valid analytical SE and bootstrap SE via the placebo IF. See
   ``docs/methodology/REGISTRY.md`` for the full contract.

.. note::

   ``ChaisemartinDHaultfoeuille`` supports ``survey_design`` with pweight
   and strata/PSU/FPC via Taylor Series Linearization. Replicate weights
   are not yet supported.

Synthetic DiD
~~~~~~~~~~~~~

Use :class:`~diff_diff.SyntheticDiD` when:

- You have few treated units but many control units
- Pre-treatment fit between treated and control is poor
- You want to construct a weighted synthetic control

.. code-block:: python

   from diff_diff import SyntheticDiD, generate_did_data

   # SyntheticDiD requires block treatment (constant within units)
   block_data = generate_did_data(n_units=40, n_periods=10, treatment_effect=2.0)
   sdid = SyntheticDiD()
   results = sdid.fit(block_data, outcome='outcome', unit='unit',
                      time='period', treatment='treated')

   # View the unit weights
   print(results.unit_weights)

Continuous Treatment
~~~~~~~~~~~~~~~~~~~~

Use :class:`~diff_diff.ContinuousDiD` when:

- Treatment varies in **intensity or dose** (e.g., subsidy amount, hours of training)
- You want to estimate how effects change with treatment dose
- You need the full dose-response curve, not just a single average effect
- Staggered adoption where units receive different treatment levels

.. note::

   Dose-response curves ATT(d) and ACRT(d) require **Strong Parallel Trends (SPT)**.
   Under standard PT only the binarized ATT\ :sup:`loc` is identified.
   Data must include an untreated group (D = 0), a balanced panel, and
   time-invariant dose (each unit's dose is fixed across periods).

.. code-block:: python

   from diff_diff import ContinuousDiD, generate_continuous_did_data

   data = generate_continuous_did_data(n_units=200, seed=42)

   est = ContinuousDiD(n_bootstrap=199, seed=42)
   results = est.fit(data, outcome='outcome', unit='unit',
                     time='period', first_treat='first_treat',
                     dose='dose')

   # Overall effect and dose-response curve (always computed by fit)
   print(f"Overall ATT: {results.overall_att:.3f}")
   att_curve = results.dose_response_att.to_dataframe()

Universal Rollout / No Untreated Control
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use :class:`~diff_diff.HeterogeneousAdoptionDiD` when:

- **Every unit is treated at the post period** (universal-rollout policy,
  industry-wide tariff change, simultaneous launch into all markets)
- Treatment **intensity (dose) varies across units**, but no genuinely
  untreated control group exists to anchor a standard DiD contrast
- :class:`~diff_diff.ContinuousDiD` is unavailable because its untreated-group
  requirement (``D = 0``) is violated

The estimator implements de Chaisemartin, Ciccia, D'Haultfoeuille and Knau
(2026, arXiv:2405.04465v6) and resolves to one of two estimands depending on
the dose support:

- **Design 1' (QUG case, ``d_lower = 0``)** identifies the **Weighted Average
  Slope (WAS)** under the Quasi-Untreated-Group assumption (units with the
  smallest dose serve as the comparison anchor). The shipped result class
  exposes ``target_parameter == "WAS"``.
- **Design 1 (no QUG, ``d_lower > 0``)** identifies ``WAS_{d_lower}`` under
  Assumption 6, or sign identification only under Assumption 5; neither
  additional assumption is testable via pre-trends. Result class exposes
  ``target_parameter == "WAS_d_lower"``.

The dose-distribution path is auto-detected. Run
:func:`~diff_diff.did_had_pretest_workflow` to vet the identifying assumptions
before estimation; see :doc:`api/had` for the full API and SE-regime contract.

.. code-block:: python

   import numpy as np
   import pandas as pd
   from diff_diff import HeterogeneousAdoptionDiD, did_had_pretest_workflow

   # Build a HAD-shape panel: D=0 in pre-periods (t < F), D > 0 only at F+.
   rng = np.random.default_rng(42)
   G, F, T = 200, 4, 5
   doses = rng.beta(0.5, 1.0, size=G)
   rows = []
   for g in range(G):
       for t in range(1, T + 1):
           y = (rng.normal()
                + (doses[g] + doses[g] ** 2) * (t >= F)
                + rng.normal(0, 0.5))
           d = doses[g] if t >= F else 0.0
           rows.append({'unit': g, 'period': t, 'y': y, 'dose': d})
   had_data = pd.DataFrame(rows)

   pretests = did_had_pretest_workflow(had_data, outcome='y', unit='unit',
                                       time='period', dose='dose')

   est = HeterogeneousAdoptionDiD()
   results = est.fit(had_data, outcome='y', unit='unit',
                     time='period', dose='dose')

   # Event-study results: per-horizon WAS at each event time
   for e, att in zip(results.event_times, results.att):
       print(f"  e={e}: {att:.3f}")

Efficient DiD
~~~~~~~~~~~~~

Use :class:`~diff_diff.EfficientDiD` when:

- You have staggered adoption and want **maximum statistical efficiency** on the no-covariate path
- You believe parallel trends holds across all pre-treatment periods (PT-All)
- You want tighter confidence intervals than Callaway-Sant'Anna
- You need a formal efficiency benchmark for comparing estimators

.. note::

   EfficientDiD supports covariate adjustment via a doubly-robust path with
   all nuisances estimated nonparametrically: sieve-based propensity score
   ratios, a sieve outcome regression (polynomial basis, AIC/BIC order
   selection), and a kernel-smoothed conditional covariance. The DR property
   gives consistency if either the outcome regression or the PS is correctly
   specified, and the covariate path attains the semiparametric efficiency
   bound asymptotically under the paper's regularity conditions (a growing
   sieve; degree 1 reproduces a linear working model, and ``sieve_k_max=1``
   forces all covariate-path sieves to degree 1). Pass column
   names to the ``covariates`` parameter on ``fit()``. See
   ``docs/methodology/REGISTRY.md`` for the full contract.

.. code-block:: python

   from diff_diff import EfficientDiD

   edid = EfficientDiD(pt_assumption="all")  # or "post" for post-treatment CS match
   results = edid.fit(data, outcome='y', unit='unit_id',
                      time='period', first_treat='first_treat')
   results.print_summary()

Sun-Abraham
~~~~~~~~~~~

Use :class:`~diff_diff.SunAbraham` when:

- You have staggered adoption and want an interaction-weighted event study
- You want to decompose effects by cohort and relative time
- You need a regression-based complement to Callaway-Sant'Anna

Sun & Abraham (2021) uses a saturated TWFE regression with cohort x relative-time
interactions, then aggregates cohort-specific effects using interaction weights.

.. code-block:: python

   from diff_diff import SunAbraham

   sa = SunAbraham(control_group='never_treated')
   results = sa.fit(data, outcome='y', unit='unit_id',
                    time='period', first_treat='first_treat')
   results.print_summary()

.. note::

   Running both Sun-Abraham and Callaway-Sant'Anna provides a useful robustness
   check. Both are consistent under heterogeneous treatment effects.

Imputation DiD
~~~~~~~~~~~~~~

Use :class:`~diff_diff.ImputationDiD` when:

- You have staggered adoption with homogeneous treatment effects
- You want shorter confidence intervals than Callaway-Sant'Anna (~50% shorter)
- You need imputed counterfactual outcomes for treated observations

Borusyak, Jaravel & Spiess (2024) estimate unit + time FE on untreated observations,
impute counterfactual Y(0) for treated observations, then aggregate.

.. code-block:: python

   from diff_diff import ImputationDiD

   imp = ImputationDiD()
   results = imp.fit(data, outcome='y', unit='unit_id',
                     time='period', first_treat='first_treat')
   results.print_summary()
   results.aggregate('event_study').summary()  # post-fit aggregation

.. note::

   Under homogeneous effects, ImputationDiD is semiparametrically efficient.
   If you suspect heterogeneous effects across cohorts, prefer Callaway-Sant'Anna.

Two-Stage DiD
~~~~~~~~~~~~~

Use :class:`~diff_diff.TwoStageDiD` when:

- You want the same point estimates as ImputationDiD with a different variance estimator
- You prefer the GMM sandwich variance that accounts for first-stage uncertainty
- You want a single ATT or an event study from a two-stage procedure

Gardner (2022) estimates FE on untreated obs (stage 1), residualizes all outcomes,
then regresses residuals on treatment indicators (stage 2).

.. code-block:: python

   from diff_diff import TwoStageDiD

   ts = TwoStageDiD()
   results = ts.fit(data, outcome='y', unit='unit_id',
                    time='period', first_treat='first_treat')
   results.print_summary()
   results.aggregate('event_study').summary()  # post-fit aggregation

.. note::

   Point estimates are identical to ImputationDiD; the key difference is the
   variance estimator (GMM sandwich vs. conservative clustered).

Stacked DiD
~~~~~~~~~~~

Use :class:`~diff_diff.StackedDiD` when:

- You have staggered adoption and want a sub-experiment approach
- You want to avoid forbidden comparisons in TWFE by construction
- You need corrective Q-weights for unbiased stacked estimation

Wing, Freedman & Hollingsworth (2024) create one sub-experiment per adoption cohort
with clean controls and apply Q-weights to reweight the stacked regression.

.. code-block:: python

   from diff_diff import StackedDiD

   stk = StackedDiD(kappa_pre=2, kappa_post=3)
   results = stk.fit(data, outcome='y', unit='unit_id',
                     time='period', first_treat='first_treat')
   results.print_summary()

   # The event-study surface is always computed (3.9); view it post-fit:
   es = results.aggregate('event_study')

.. note::

   The trimmed aggregate ATT may exclude early or late cohorts whose event
   windows do not fit in the data. Check ``results.trimmed_groups``.

TROP
~~~~

Use :class:`~diff_diff.TROP` when:

- You suspect interactive fixed effects (factor confounding)
- Standard parallel trends may not hold due to unobserved factors
- You want triple robustness: factor model + unit weights + time weights

Athey, Imbens, Qu & Viviano (2025) combine nuclear norm regularization,
exponential unit distance weights, and time decay weights with LOOCV tuning.

.. code-block:: python

   from diff_diff import TROP

   trop = TROP(n_bootstrap=200)
   results = trop.fit(data, outcome='y', treatment='treated',
                      unit='unit_id', time='period')
   results.print_summary()

.. note::

   TROP is computationally intensive. Use ``method='global'`` for faster
   estimation at the cost of some flexibility vs. ``method='local'``.

LWDiD (Lee & Wooldridge)
~~~~~~~~~~~~~~~~~~~~~~~~

**When to use**: Panel data where unit-specific rolling transformations
(demeaning or detrending) can remove pre-treatment heterogeneity, combined
with flexible cross-sectional treatment effect estimation (RA, IPW, IPWRA,
or PSM). Particularly suited when you want a transformation-based
alternative to propensity-score reweighting under staggered adoption.

**Key features**:

- Converts panel DiD into cross-sectional estimation via unit-specific
  transformations (demean or detrend) applied to pre-treatment outcomes
- Supports both common timing and staggered adoption designs
  (never-treated / not-yet-treated controls)
- Doubly-robust estimation (``estimation_method='dr'``) with
  influence-function inference (``vcov_type='hc1'``; the ``reg`` path
  additionally offers classical/HC2/HC3); cluster-robust inference via
  the constructor's ``cluster=`` parameter
- Built-in specification robustness: compare demean vs detrend as an
  informal pre-test for sensitivity to trend assumptions

**vs TWFE**: LWDiD explicitly handles heterogeneous treatment effects;
the transformation removes unit fixed effects prior to estimation, avoiding
the negative-weighting problem under treatment effect heterogeneity.

**vs Callaway-Sant'Anna**: LWDiD uses rolling transformations rather than
propensity-score reweighting for staggered designs, offering a different
identification strategy with analytical (non-bootstrap) inference.

**Example**::

    from diff_diff import LWDiD
    data['size'] = data['id'] % 5  # unit-constant covariate for the DR propensity model
    est = LWDiD(rolling='demean', estimation_method='dr', cluster='state')
    results = est.fit(data, outcome='y', unit='id', time='time',
                      treatment='treated', first_treat='first_treat',
                      covariates=['size'])  # DR needs covariates; without them it reduces to RA

DMLDiD (Chang 2020)
~~~~~~~~~~~~~~~~~~~

**When to use**: Staggered (or 2-period) designs — panel data or declared
repeated cross sections (``panel=False``) — where parallel trends is
plausible only CONDITIONAL on covariates and the covariate adjustment
must be flexible or high-dimensional — double/debiased machine learning
(DML2 cross-fitting + Neyman-orthogonal scores) makes the ATT first-order
insensitive to the nuisance learners' regularization bias.

**Key features**:

- Per-(g, t) cell Chang (2020) estimation on the Callaway-Sant'Anna cell
  architecture — Case 1 (outcome changes) on panel data, Case 2 (level
  outcomes with the λ-corrected variance) on declared repeated cross
  sections; the 2-period design is the degenerate single-cell case
- Configurable nuisance learners: string names (``'logit'``; ``'linear'``,
  ``'ridge'``, ``'sieve'``) or any object with ``fit``/``predict``
  (``predict_proba``) — sklearn estimators plug in directly;
  :class:`~diff_diff.SieveLearner` is the exported adaptive-degree option
- Covariates are REQUIRED (the no-covariates case routes to
  :class:`~diff_diff.CallawaySantAnna`)
- Post-fit aggregation (``results.aggregate('event_study'/'group'/
  'simple')``, plus ``'total'`` on panel fits — RCS fits fail ``total``
  closed) with sup-t bands and bootstrap replay; HonestDiD /
  PreTrendsPower consume the event-study container
- Analytical augmented-score inference: the panel lane is anchored to
  DoubleML at machine precision (committed parity spikes); the RCS lane
  has no DoubleML oracle (different score) and is validated by equation
  fixtures + a committed characterization spike

**vs Callaway-Sant'Anna**: same cell architecture and aggregation surface;
DMLDiD replaces CS's parametric nuisances with cross-fitted ML learners —
prefer it when the covariate relationship is nonlinear/high-dimensional,
prefer CS otherwise (fewer moving parts, replicate-weight support). Both
handle declared repeated cross sections via ``panel=False``, and both
carry pweight survey designs and ``cluster=`` there (DMLDiD's survey
lane is a documented library extension of Chang's i.i.d. theory;
replicate-weight designs stay CS-only).

**Example**::

    from diff_diff import DMLDiD
    est = DMLDiD(outcome_learner='sieve', seed=42)
    results = est.fit(data, outcome='y', unit='id', time='time',
                      first_treat='first_treat', covariates=['x1', 'x2'])
    print(results.aggregate('event_study').to_dataframe())

Bacon Decomposition
~~~~~~~~~~~~~~~~~~~

Use :class:`~diff_diff.BaconDecomposition` when:

- You want to **diagnose** whether TWFE is biased in your staggered setting
- You need to see which 2x2 comparisons drive the TWFE estimate
- You want to check whether later-vs-earlier or already-treated-as-control comparisons carry substantial weight

Goodman-Bacon (2021) decomposes the TWFE estimate into a weighted average of
all 2x2 DiD comparisons and their weights.

.. code-block:: python

   from diff_diff import BaconDecomposition, plot_bacon

   bacon = BaconDecomposition()
   results = bacon.fit(data, outcome='y', unit='unit_id',
                       time='period', first_treat='first_treat')
   results.print_summary()

   # Visualize the decomposition
   plot_bacon(results)

.. note::

   This is a diagnostic tool, not an estimator. If the decomposition reveals
   problematic weights, switch to Callaway-Sant'Anna or another robust estimator.

Common Pitfalls
---------------

1. **Using TWFE with staggered adoption**

   TWFE estimates a weighted average of all 2x2 comparisons, including
   "forbidden" comparisons where already-treated units serve as controls.
   This can lead to severe bias, even negative weights on treatment effects.

   *Solution*: Use CallawaySantAnna for staggered designs.

2. **Ignoring treatment effect heterogeneity**

   If treatment effects vary by cohort (when units are treated) or over time
   (dynamic effects), aggregated estimators may be misleading.

   *Solution*: Use CallawaySantAnna and examine ATT(g,t) and event study plots.

3. **Failing to test parallel trends**

   The parallel trends assumption is untestable in the post-period but can
   be assessed using pre-treatment data.

   *Solution*: Use :func:`~diff_diff.check_parallel_trends` and
   :class:`~diff_diff.HonestDiD` for sensitivity analysis.

4. **Inappropriate clustering**

   Standard errors should typically be clustered at the level of treatment
   assignment (often the unit level).

   *Solution*: Always specify ``cluster`` for panel data.

Standard Error Methods
----------------------

Different estimators compute standard errors differently. Understanding these
differences helps interpret results and choose appropriate inference.

.. list-table::
   :header-rows: 1
   :widths: 20 25 55

   * - Estimator
     - Default SE Method
     - Details
   * - ``DifferenceInDifferences``
     - HC1 (heteroskedasticity-robust)
     - Uses White's robust SEs by default. Specify ``cluster`` for cluster-robust SEs. Use ``inference='wild_bootstrap'`` (with ``cluster=`` — required) for few clusters (<50).
   * - ``TwoWayFixedEffects``
     - Cluster-robust (unit level)
     - Always clusters at unit level after within-transformation (static AND event-study mode). Specify ``cluster`` to override. Use ``inference='wild_bootstrap'`` for few clusters (static mode only; event-study mode raises).
   * - ``MultiPeriodDiD`` (deprecated 3.9)
     - HC1 (heteroskedasticity-robust)
     - Same as basic DiD. Cluster-robust available via ``cluster``. Wild bootstrap not yet supported for multi-coefficient inference.
   * - ``CallawaySantAnna``
     - Analytical (influence function)
     - Uses influence-function SEs with WIF adjustment by default. Set ``n_bootstrap=999`` for multiplier bootstrap inference (weight types: ``rademacher``, ``mammen``, ``webb``).
   * - ``SyntheticDiD``
     - Placebo, paper-faithful refit bootstrap, or jackknife
     - Default uses placebo-based variance (``variance_method="placebo"``). Set ``variance_method="bootstrap"`` for paper-faithful Algorithm 2 bootstrap (re-estimates ω and λ via Frank-Wolfe per draw; ~5–30× slower than placebo, panel-size dependent). Both methods use ``n_bootstrap`` replications (default 200). ``variance_method="jackknife"`` is also available.
   * - ``ContinuousDiD``
     - Analytical (influence function)
     - Uses influence-function-based SEs by default. Use ``n_bootstrap=199`` (or higher) for multiplier bootstrap inference with proper CIs.
   * - ``HeterogeneousAdoptionDiD``
     - Path-dependent (CCT-2014 / 2SLS / Binder TSL)
     - Two SE regimes per :doc:`api/had`. **Unweighted**: continuous-dose paths use the CCT-2014 robust SE from the in-house ``lprobust`` port; mass-point uses a 2SLS sandwich. **``survey_design=SurveyDesign(weights="col", ...)``** (the sole weighting entry as of the 3.7.0 ``survey=`` / ``weights=`` removal): both paths compose Binder (1983) Taylor-series linearization (``variance_formula="survey_binder_tsl"`` / ``"survey_binder_tsl_2sls"``); the mass-point survey path rejects ``vcov_type="classical"`` (requires ``vcov_type="hc1"``), and ``survey_design=`` + ``cluster=`` is rejected outright (route weighted clustering via ``SurveyDesign(weights=, psu=)``; a bare ``cluster=`` gives unweighted CR1). Per-horizon CIs are pointwise; sup-t bands available on the event-study path via ``cband=True`` whenever ``survey_design=`` or ``cluster=`` is supplied.
   * - ``RegressionDiscontinuity``
     - Robust bias-corrected (CCT 2014, NN variance)
     - Sharp, fuzzy, and covariate-adjusted RD with rdrobust-4.0.0-parity inference (fuzzy via ``fit(..., takeup=...)``: local Wald ratio with a linearized bias correction, first-stage block, and a weak-first-stage warning; covariates via ``fit(..., covariates=[...])``: same estimand, precision only, covariate-aware bandwidths). Canonical ``att``/``se``/``conf_int`` are the ROBUST bias-corrected row (``att`` = bias-corrected estimate, CI centered on it); the conventional estimate rdrobust prints as its headline is ``att_conventional`` with its own inference row. Only ``vcov_type="nn"`` in this release; cluster-robust RD variance is a documented follow-up.
   * - ``SunAbraham``
     - Cluster-robust (unit level)
     - Clusters at unit level by default. Specify ``cluster`` to override. Use ``n_bootstrap`` for pairs bootstrap inference.
   * - ``ImputationDiD``
     - Conservative clustered (Theorem 3)
     - Uses conservative clustered variance from Borusyak et al. Theorem 3, clustered at unit level. Use ``n_bootstrap`` for multiplier bootstrap.
   * - ``TwoStageDiD``
     - GMM sandwich (clustered)
     - Uses GMM sandwich variance accounting for first-stage estimation uncertainty, clustered at unit level. Use ``n_bootstrap`` for multiplier bootstrap.
   * - ``StackedDiD``
     - Cluster-robust (unit level)
     - Clusters at unit level by default. Set ``cluster='unit_subexp'`` for (unit, sub-experiment) clustering.
   * - ``TripleDifference``
     - Influence function (robust)
     - Uses influence-function-based SEs (inherently heteroskedasticity-robust). Specify ``cluster`` for cluster-robust SEs.
   * - ``TROP``
     - Bootstrap (n_bootstrap=200)
     - Uses unit-level block bootstrap for variance estimation. Bootstrap is always required (minimum n_bootstrap=2).
   * - ``EfficientDiD``
     - Analytical (EIF-based)
     - Uses efficient influence function SE = sqrt(mean(EIF^2) / n). Use ``n_bootstrap`` for multiplier bootstrap.
   * - ``LWDiD``
     - HC1 on the collapsed regression
     - ``estimation_method='reg'`` accepts ``classical`` (exact small-sample t with ``n_bootstrap=0`` — a positive ``n_bootstrap`` replaces the headline inference with the unit/cluster bootstrap — valid down to one treated unit; causal only under the LWDiD identifying assumptions — no anticipation, parallel trends or its heterogeneous-linear-trends variant for ``rolling='detrend'``, and overlap — and exact only under the classical error assumptions; the complete contract, including the covariate-design fallback, is in :doc:`api/lwdid` Key Assumptions / Small-Sample Inference; staggered aggregates other than the eligible never-treated composite use large-sample IF inference instead), ``hc1``, ``hc2``, ``hc3``; ``cluster=`` gives CR1 (composes with ``hc1`` only). ``ipw``/``dr`` use the influence-function variance (``hc1`` only). Post-fit ``randomization_test()`` and ``wild_cluster_bootstrap()`` on common-timing ``reg`` fits. ``psm`` reports NaN inference (no valid matching variance implemented).
   * - ``BaconDecomposition``
     - N/A (diagnostic)
     - Diagnostic tool only; does not produce standard errors.

**Recommendations by sample size:**

- **Large samples (>= 50 clusters)**: Cluster-robust SEs are reliable (the asymptotic
  approximation holds; the 50-cluster threshold is the diff-diff convention used
  throughout the guides)
- **Small samples (clusters < 50)**: Use wild cluster bootstrap (``inference='wild_bootstrap'`` with ``cluster=`` — ``TwoWayFixedEffects`` auto-clusters at unit level)
- **Very few clusters (< 10)**: Use Webb 6-point distribution (``bootstrap_weights='webb'``)

**Common pitfall:** Forgetting to cluster when units are observed multiple times.
For panel data, always cluster at the unit level unless you have a strong reason not to.

.. code-block:: python

   from diff_diff import DifferenceInDifferences, generate_did_data

   panel = generate_did_data(n_units=200, n_periods=10, treatment_effect=2.0)

   # Good: Cluster at unit level for panel data
   did = DifferenceInDifferences(cluster='unit')
   results = did.fit(panel, outcome='outcome', treatment='treated',
                     post='post')

   # Better for few clusters: Wild bootstrap
   did = DifferenceInDifferences(inference='wild_bootstrap', cluster='unit')
   results = did.fit(panel, outcome='outcome', treatment='treated',
                     post='post')

When in Doubt
-------------

If you're unsure which estimator to use:

1. **Start with CallawaySantAnna** - It's valid even for non-staggered designs
   and provides the most flexible output (group-time effects, aggregations)

2. **Check for heterogeneity** - Plot event studies to see if effects vary

3. **Run sensitivity analysis** - Use HonestDiD to assess robustness

4. **Compare estimators** - If results differ substantially across estimators,
   investigate why (often reveals violations of assumptions)

5. **Using survey data?** - Pass a ``SurveyDesign`` to ``fit()`` for design-based
   variance estimation. See the :ref:`survey-design-support` section below for
   the compatibility matrix, and the `survey tutorial <https://github.com/igerber/diff-diff/blob/main/docs/tutorials/16_survey_did.ipynb>`_
   for a full walkthrough.

.. _survey-design-support:

Survey Design Support
---------------------

Most estimators support an optional ``survey_design`` parameter in ``fit()``
(``SyntheticControl`` accepts the parameter but raises ``NotImplementedError``;
``LWDiD`` accepts no ``survey_design`` parameter at all —
passing it raises ``TypeError``).
Pass a :class:`~diff_diff.SurveyDesign` object to get design-based variance
estimation. The depth of support varies by estimator and variance method:

.. note::

   If your data starts as **individual-level survey microdata** (e.g., BRFSS,
   ACS, CPS, NHANES respondent records), use :func:`~diff_diff.aggregate_survey`
   as a preprocessing step. It pools microdata into geographic-period cells and
   returns a pre-configured :class:`~diff_diff.SurveyDesign`. By default, the
   returned design uses ``weight_type="pweight"`` (unit-constant population
   weights), which is compatible with **all** survey-capable
   estimators in the matrix below. Pass ``second_stage_weights="aweight"`` for
   precision weights (inverse variance) if you prefer efficiency-weighted
   estimates - this mode is limited to estimators marked **Full**.
   See :doc:`api/prep` for the API reference.

.. list-table::
   :header-rows: 1
   :widths: 25 12 18 18 18

   * - Estimator
     - Weights
     - Strata/PSU/FPC
     - Replicate Weights
     - Survey Bootstrap
   * - ``DifferenceInDifferences``
     - Full
     - Full
     - Full
     - --
   * - ``TwoWayFixedEffects``
     - Full
     - Full
     - Full
     - --
   * - ``MultiPeriodDiD``
     - Full
     - Full
     - Full
     - --
   * - ``CallawaySantAnna``
     - pweight only
     - Full
     - Full
     - Multiplier at PSU
   * - ``ChaisemartinDHaultfoeuille``
     - pweight only
     - Full (TSL)
     - Full (analytical)
     - Group-level (warning)
   * - ``TripleDifference`` (2x2x2 mode)
     - pweight only
     - Full
     - Full (analytical)
     - --
   * - ``TripleDifference`` (staggered mode, ``first_treat=``)
     - pweight only
     - Full
     - Full
     - Multiplier at PSU
   * - ``SunAbraham``
     - Full
     - Full
     - Full
     - Rao-Wu rescaled
   * - ``StackedDiD``
     - pweight only
     - Full (pweight only)
     - Full
     - --
   * - ``ImputationDiD``
     - pweight only
     - Full
     - Full (analytical)
     - Multiplier at PSU
   * - ``TwoStageDiD``
     - pweight only
     - Full
     - Full (analytical)
     - Multiplier at PSU
   * - ``ContinuousDiD``
     - Full
     - Full
     - Full (analytical)
     - Multiplier at PSU
   * - ``HeterogeneousAdoptionDiD``
     - pweight only
     - Full (Binder TSL)
     - --
     - Multiplier (event-study, ``cband=True`` only)
   * - ``RegressionDiscontinuity``
     - N/A (no survey support)
     - N/A
     - --
     - --
   * - ``EfficientDiD``
     - Full
     - Full
     - Full (analytical)
     - Multiplier at PSU
   * - ``SyntheticDiD``
     - pweight only
     - Full (method-specific)
     - --
     - Hybrid pairs-bootstrap + Rao-Wu rescaled (bootstrap only)
   * - ``SyntheticControl``
     - --
     - --
     - --
     - --
   * - ``TROP``
     - pweight only
     - Via bootstrap
     - --
     - Rao-Wu rescaled
   * - ``WooldridgeDiD``
     - Full (pweight only)
     - Full (analytical)
     - --
     - --
   * - ``LPDiD``
     - pweight only
     - Full (Binder TSL)
     - --
     - --
   * - ``LWDiD``
     - N/A (no survey support)
     - N/A
     - --
     - --
   * - ``DMLDiD``
     - Full (pweight only)
     - Full (TSL; df = ``n_PSU - n_strata``)
     - --
     - Multiplier (PSU)
   * - ``ChangesInChanges`` / ``QDiD``
     - --
     - --
     - --
     - --
   * - ``SpilloverDiD``
     - pweight only
     - Full (Binder TSL + Conley)
     - --
     - --
   * - ``BaconDecomposition``
     - Diagnostic
     - Diagnostic
     - --
     - --

**Legend:**

- **Full**: All weight types (pweight/fweight/aweight) + strata/PSU/FPC + Taylor Series Linearization variance
- **Full (pweight only)**: Full TSL with strata/PSU/FPC, but only ``pweight`` accepted (``fweight``/``aweight`` rejected because composition changes weight semantics)
- **Via bootstrap**: Strata/PSU/FPC supported only with bootstrap variance (``TROP``, which uses bootstrap by default)
- **Full (method-specific)**: ``SyntheticDiD`` supports strata/PSU/FPC on all three variance methods via method-specific survey paths — see the note below and the ``Note (survey support matrix)`` in REGISTRY.md §SyntheticDiD
- **pweight only** (Weights column): Only ``pweight`` accepted; ``fweight``/``aweight`` raise an error
- **Diagnostic**: Weighted descriptive statistics only (no inference)
- **--**: Not supported

.. note::

   ``SyntheticDiD`` supports survey designs — both pweight-only and full
   strata/PSU/FPC — on all three variance methods, each via a
   method-specific path: ``bootstrap`` composes a hybrid pairs-bootstrap
   with per-draw Rao-Wu rescaled weights fed into a weighted Frank-Wolfe
   re-estimation of ω and λ; ``placebo`` switches to stratified
   permutation (pseudo-treated draws within strata containing treated
   units) with weighted-FW re-estimation, and FPC is a documented no-op
   for the permutation test; ``jackknife`` switches to PSU-level
   leave-one-out with stratum aggregation (Rust & Rao 1996).
   Replicate-weight designs are rejected. See the
   ``Note (survey support matrix)`` and the per-method composition notes
   in REGISTRY.md §SyntheticDiD.

For the full walkthrough with code examples, see the
`survey tutorial <https://github.com/igerber/diff-diff/blob/main/docs/tutorials/16_survey_did.ipynb>`_.
For deferred work and remaining limitations, see ``docs/survey-roadmap.md``.
