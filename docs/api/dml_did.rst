DMLDiD — Double/Debiased Machine Learning DiD
=============================================

Chang (2020)'s double/debiased machine learning (DML) estimator for
Difference-in-Differences with covariates, extended to staggered adoption:
each Callaway-Sant'Anna style group-time cell :math:`(g, t)` is estimated
as a 2-period Chang problem with cross-fitted machine-learning nuisance
functions and a Neyman-orthogonal score, so the ATT estimate is
first-order insensitive to the regularization bias of the nuisance
learners. ``panel=True`` (the default) runs Case 1 (repeated outcomes) on
panel data; ``panel=False`` runs Case 2 (repeated cross sections) on
declared cross-sectional data — one observation per row. The classic
2-period design is the degenerate single-cell case of either lane.

Identification follows Abadie (2005): CONDITIONAL parallel trends — the
untreated potential-outcome trend is parallel across treated and control
units only after conditioning on covariates :math:`X`. Chang's
contribution is valid :math:`\sqrt{n}` inference when the two nuisance
functions — the propensity score :math:`g_0(X) = P(D=1|X)` plus, on panel
fits, the control outcome-change regression
:math:`\ell_0(X) = E[\Delta Y | X, D=0]` (Case 1) or, on RCS fits, the
control level regression
:math:`\ell_{20}(X) = E[(T - \lambda) Y | X, D=0]` (Case 2) — are
estimated by machine learning under DML2 cross-fitting — PROVIDED the
learners satisfy Chang's rate conditions (Assumption 3.1(f) for Case 1 /
3.2(h) for Case 2: each
nuisance at :math:`o_p(n^{-1/4})` in the :math:`L_2` norm, plus a product
remainder bound). Cross-fitting removes overfitting bias but cannot
substitute for the rate conditions, and the reported fold losses do not
verify them; slow learners silently invalidate normal inference (see the REGISTRY
DMLDiD "First-stage rate condition" assumption bullet).

Covariates are **required**: Chang's estimator exists for the
conditional-on-covariates setting. For an unconditional staggered DiD use
:class:`~diff_diff.CallawaySantAnna` (with or without covariates).

.. module:: diff_diff.dml_did

Methodology
-----------

**Per-cell orthogonal score (Chang 2020, Equation 3.1).** For cell
:math:`(g, t)` with base period :math:`b` selected positionally (the CS /
R ``did`` rule), let :math:`\Delta Y_i = Y_{it} - Y_{ib}`, treated
indicator :math:`D_i = 1\{\text{cohort}_i = g\}`, and cell treated share
:math:`\hat p`. The uncentered score summand is:

.. math::

   \text{summand}_i = \frac{D_i - \hat g(X_i)\,(1 - D_i)/(1 - \hat g(X_i))}{\hat p}\,\bigl(\Delta Y_i - \hat\ell(X_i)\bigr)

and :math:`\widehat{ATT}(g,t)` is its sample mean over the cell. Both
nuisances are cross-fitted OUT-OF-FOLD (DML2, K folds, per-cell
D-stratified assignment): the propensity by the classifier learner, the
outcome-change regression by the regressor learner **trained on the cell's
control units only** (Chang's :math:`I_{kz}^c`).

**Variance (Chang 2020, Theorem 2).** The augmented score
:math:`\bar\psi_i = \text{summand}_i - D_i \hat\theta / \hat p` folds in
the finite-dimensional treated-share correction; the per-cell standard
error is the plug-in :math:`\sqrt{\overline{\bar\psi^2} / n}`. This exact
object was matched to DoubleML at machine precision in the committed
parity spikes (``benchmarks/doubleml/``).

**Case 2 — repeated cross sections (Chang 2020, Equation 3.2;**
``panel=False``\ **).** The cell pools the two periods' rows (post indicator
:math:`T_i = 1\{\text{time}_i = t\}`) and scores LEVEL outcomes:

.. math::

   \text{summand}_i = \frac{D_i - \hat g(X_i)}{\hat p\,\hat\lambda(1-\hat\lambda)\,(1 - \hat g(X_i))}\,\bigl((T_i - \hat\lambda)Y_i - \hat\ell_2(X_i)\bigr)

with :math:`\hat\lambda = \text{mean}(T)` the post-period sampling share and
:math:`\hat\ell_2` the SINGLE cross-fitted control-only regression of
:math:`(T - \hat\lambda)Y` on :math:`X` (Chang's :math:`I_{kz}^c`) — one
regression, deliberately different from the Sant'Anna-Zhao/DoubleML
four-regression RCS score. The Theorem 2 variance carries BOTH
finite-dimensional corrections: the treated-share fold-in plus an explicit
:math:`\hat G_{2\lambda}(T_i - \hat\lambda)` term (the λ-correction the
paper's proof structure warns is easy to omit), with
:math:`\hat G_{2\lambda}` the sample mean of the closed-form
:math:`\partial_\lambda \psi_2`. Folds are stratified on the four
:math:`D \times T` classes so every training fold carries control rows in
both periods. RCS aggregation weights are FIXED cohort row masses (the
CS-RCS convention, keeping the variance the influence function of the
reported aggregate); ``aggregate('total')`` is unavailable on RCS fits.

**Bad controls (Caetano, Callaway, Payne & Sant'Anna 2026;**
``fit(..., bad_control=, bad_control_covariates=)``\ **).** A "bad control"
is a time-varying covariate :math:`X_t` that treatment can affect.
Conditioning on it at :math:`t` biases DiD; conditioning only on its
pre-treatment value (pass it in ``covariates`` - the panel lane reads
covariates at the cell's base period) is the paper's Approach 1. This
lane implements the paper's covariate-unconfoundedness approach: parallel
trends conditional on the bad control's UNTREATED path, with the bad
control's untreated evolution identified given its base-period value, the
optional ``bad_control_covariates`` :math:`W` (the outcome name means the
base-period outcome, the paper's Remark 5 recommendation), and the
covariates :math:`Z`. Per cell, with :math:`R = [X_t, X_b, Z]` and
:math:`S = [X_b, W, Z]`, the Neyman-orthogonal doubly-robust score
(paper Equation 10) is

.. math::

   \text{summand}_i = \frac{D_i}{\hat\pi}\bigl(\Delta Y_i - \hat\nu(S_i)\bigr)
   - \frac{1 - D_i}{\hat\pi}\Bigl[\bigl(\hat m(R_i) - \hat\nu(S_i)\bigr)\frac{\hat p(S_i)}{1 - \hat p(S_i)}
   + \bigl(\Delta Y_i - \hat m(R_i)\bigr)\hat\omega(R_i)\Bigr]

with four cross-fitted nuisances: the control outcome-change regression
:math:`\hat m` on :math:`R`, the propensity :math:`\hat p` on :math:`S`, and
two NESTED second-stage regressions - :math:`\hat\nu` (the first-stage
predictions regressed on :math:`S` among training controls) and
:math:`\hat\omega` (the propensity odds regressed on :math:`R`, clipped to
:math:`[0, (1-\text{trim})/\text{trim}]` with a warning). With the parametric
built-in learners the nested targets are the paper's own plug-in; with
``ridge`` / ``sieve`` / user learners the training complement is split in
half (footnote 9). The variance is the same augmented-score plug-in as the
plain lane. Each cell also reports :math:`\widehat{ATT}_X(g,t)`, the
paper's Remark 6 pre-test (the effect of treatment on the bad control
itself, an AIPW mean-effect diagnostic with its own analytical SE), via
``results.bad_control_summary()``: pre-period rows (:math:`t < g`) assess
the identifying assumptions MP-5 / MP-8 and should be zero (a nonzero
value flags a possible violation), post-period rows (:math:`t \ge g`) are
the Condition-2 check (a nonzero value is evidence treatment affects the
covariate; a zero value does not establish the converse). Restrictions: panel lane only, bare
``cluster=`` only (``survey_design=`` raises), ``anticipation=0`` and
``base_period='varying'`` only. The headline ``att`` keeps the CS
"simple" weighting, not the paper's Remark 4 overall. Validated against
the authors' R package ``badcontrols`` as a tolerance-based black box
(``tests/test_dml_did_bad_controls_parity.py``) plus the paper's
Supplementary DGPs 1 and 4; see ``docs/methodology/REGISTRY.md`` "DMLDiD"
> "Bad-control extension (CCPS 2026)" for every convention.

**Aggregation.** ``DMLDiD`` writes the CallawaySantAnna per-cell
influence-function payload and inherits the CS aggregation and
multiplier-bootstrap machinery: event-study / group / simple aggregations
(plus total on panel fits — RCS fits fail ``total`` closed) are produced
**post-fit** via ``results.aggregate(...)``,
with sup-t uniform bands and bootstrap replay on bootstrapped fits.

See ``docs/methodology/REGISTRY.md`` "DMLDiD" for the full equations,
implementation Notes (global :math:`\hat p` convention, D-stratified
folds, pooled fold weighting, trimming policy, skip-reason vocabulary)
and the DoubleML parity anchors.

Basic usage
-----------

.. code-block:: python

   import diff_diff as dd

   est = dd.DMLDiD(
       propensity_learner="logit",
       outcome_learner="sieve",   # adaptive polynomial sieve (IC-selected)
       n_folds=5,
       seed=42,                   # reproducible fold draws
   )
   results = est.fit(
       df,
       outcome="y",
       unit="unit",
       time="time",
       first_treat="first_treat",
       covariates=["x1", "x2"],   # REQUIRED
   )
   print(results.summary())

   # Post-fit aggregation (event study with sup-t bands on bootstrapped fits)
   es = results.aggregate("event_study")
   print(es.to_dataframe())

Bad control (Caetano et al. 2026): a covariate treatment can affect, such
as an occupation or earnings measure, is passed as ``bad_control`` rather
than in ``covariates``; ``bad_control_covariates=[outcome]`` conditions the
bad control's evolution on the base-period outcome (Remark 5), and the
explicit ``control_group="not_yet_treated"`` is the paper's Proposition 2
comparison group:

.. code-block:: python

   from diff_diff import DMLDiD

   data = data.assign(
       occupation=lambda d: 0.7 * d["x1"] + 0.3 * d["x2"] + 0.5 * d["treated"] * d["post"]
   )
   bc = DMLDiD(control_group="not_yet_treated", seed=0).fit(
       data,
       outcome="y",
       unit="unit",
       time="time",
       first_treat="first_treat",
       covariates=["x3"],
       bad_control="occupation",
       bad_control_covariates=["y"],
   )
   print(bc.summary())
   print(bc.bad_control_summary())   # per-cell ATT_X(g, t): pre-period rows should be ~0 (MP-5 / MP-8);
                                     # nonzero post-period rows = treatment moves the covariate

Learner configuration
---------------------

``propensity_learner`` and ``outcome_learner`` accept a string name
(library defaults: ``"logit"``; ``"linear"``, ``"ridge"``, ``"sieve"``) or
ANY object satisfying the duck-typed learner protocol —
``fit(X, y)`` plus ``predict`` (regressor) or ``predict_proba``
(classifier). On no-design and bare-``cluster=`` fits the cross-fitting
is unweighted, so a ``sample_weight`` parameter on ``fit`` is not
required there; a DECLARED ``survey_design=`` fit passes
``sample_weight`` into both nuisances, and a user learner whose ``fit``
cannot take ``sample_weight`` by keyword is rejected up front with a
``TypeError``. A user-constructed or scikit-learn
estimator object plugs in directly; string names select library defaults
only. :class:`~diff_diff.SieveLearner` is the exported configurable
learner (``DMLDiD(outcome_learner=SieveLearner(k_max=3))``).

With ``seed`` set, fits are reproducible with the library's deterministic
built-in learners; a user-supplied STOCHASTIC learner must additionally be
seeded by the user (e.g. sklearn ``random_state``).

API Reference
-------------

.. autoclass:: diff_diff.DMLDiD
   :no-index:
   :members: fit, get_params, set_params

.. autoclass:: diff_diff.dml_did_results.DMLDiDResults
   :no-index:
   :members: aggregate, bad_control_summary, summary, to_dict, to_dataframe

.. autoclass:: diff_diff.SieveLearner
   :no-index:
   :members: fit, predict

Restrictions
------------

.. warning::

   The following restrictions apply to the current implementation:

- **Covariates required** — ``fit(covariates=None)`` or an empty list
  raises, directing to :class:`~diff_diff.CallawaySantAnna`.
- **Declared designs only** — ``panel=True`` needs one row per
  (unit, period); ``panel=False`` needs ROW-UNIQUE unit IDs (one
  observation per row) and additionally assumes STATIONARY cross-sectional
  sampling (Chang Assumption 2.3: each wave samples the same target
  population — the composition of :math:`(D, X)` is stable across waves
  while outcomes are the period-specific potential outcomes; warned at
  fit, not data-checkable).
  ``aggregate('total')`` is unavailable on RCS fits (fails closed).
  Repeated-cross-section data is typically SURVEY data (BRFSS / ACS /
  CPS) — pass a pweight :class:`~diff_diff.SurveyDesign` via
  ``survey_design=`` for weighted RCS.
- **Survey/cluster support (both lanes)** — ``survey_design=``
  (pweight-only) weights the moment kernels and passes ``sample_weight``
  into the nuisance learners; two variance lanes. Full-design TSL
  (weights/strata/PSU/FPC) switches to PSU-cohesive cross-fitting folds
  when the PSU is strictly coarser than the sampling unit and uses
  design-based variance with ``df = n_PSU - n_strata`` t-inference: PSU
  designs get the cluster-robust survey kernel per cell, strata/FPC-only
  designs use the weighted influence-function per-cell SE with the full
  design entering the AGGREGATE variances (the CallawaySantAnna
  convention). Replicate-weight designs (BRR / Fay / JK1 / JKn / SDR)
  compute per-cell AND aggregate variances by IF-reweighting the
  cross-fitted scores with ``df = rank(replicate matrix) - 1``
  t-inference (nuisances are not re-estimated per replicate); replicate
  + ``cluster=`` and replicate + ``n_bootstrap > 0`` are rejected. This
  is a documented LIBRARY EXTENSION of Chang (2020), which assumes
  i.i.d. sampling — on the weighted-λ RCS lane Theorem 2's coverage
  claim does not carry over (REGISTRY DMLDiD Notes). Bare ``cluster=``
  (constructor) keeps the kernels unweighted and affects folds, variance
  and df only. ``aggregate('total')`` fails closed on declared-survey
  fits.
- **Propensity clipping, never dropping** — fitted propensities are
  clipped to ``[pscore_trim, 1 - pscore_trim]`` after an extremeness
  warning (the paper gives no trimming rule).
- **Per-cell complete cases** — on panel fits, a unit with a
  missing/non-finite outcome or a non-finite base-period covariate at a
  cell is excluded from that cell only; on RCS fits the exclusion is
  row-level (a non-finite outcome or covariate on the row — there is no
  base-period covariate). One consolidated ``UserWarning`` reports the
  drops.
- **Degenerate cells skip loudly** — a cell that cannot be cross-fitted
  (fewer members than folds, a singleton treated/control stratum, a
  fail-closed learner error) is recorded as a NaN cell with a
  machine-readable ``skip_reason`` and reported in a consolidated
  warning; surviving cells still aggregate.
- **Event-study surface is post-fit only** — fit-time
  ``event_study_effects`` is never populated; call
  ``results.aggregate('event_study')``.
- **Bad-control lane** — ``fit(bad_control=...)`` requires
  ``panel=True``, ``anticipation=0`` and ``base_period='varying'``, and
  no ``survey_design`` (each raises ``NotImplementedError``; bare
  ``cluster=`` is supported). The bad control may not appear in
  ``covariates`` (the "include the bad control" bias); ``covariates``
  stays required. Units with a non-finite bad control at the base period
  or at ``t``, or a non-finite bad-control covariate at the base period,
  leave that cell (complete cases). ``ATT_X`` is analytical only: never
  bootstrapped or aggregated.

.. seealso::

   :class:`~diff_diff.CallawaySantAnna`
      The unconditional / low-dimensional-covariates staggered estimator
      whose (g, t) architecture, aggregation, and bootstrap DMLDiD reuses.
   :class:`~diff_diff.EfficientDiD`
      Chen, Sant'Anna & Xie (2025) efficient staggered estimation with
      sieve nuisances (fixed-form, not cross-fitted).
   :class:`~diff_diff.LWDiD`
      Lee & Wooldridge rolling-transformation DiD (cross-sectional TE
      estimation after a unit-specific transform).
