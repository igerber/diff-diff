Duration DiD (Deaner & Ku)
==========================

Causal duration analysis with difference-in-differences for a binary
**absorbing** outcome — a spell that ends (job found, subscription cancelled,
patient discharged) and stays ended — in a two-group, common-timing design.
Standard DiD on the cumulative event indicator imposes a constant gap in
event *probabilities*, which mechanically forces the survivors' hazards to
diverge (Deaner & Ku 2026, Appendix A.1). ``DurationDiD`` instead restricts
the two groups' **untreated hazards**:

- ``method="cd"`` (common dynamics): a constant additive hazard gap ``c``.
- ``method="ph"`` (proportional hazards): a constant hazard ratio ``c``.

The coefficient is fitted from the pre-treatment cumulative hazards (default:
equal weights over every eligible pre-treatment date after the baseline;
``pre_periods=`` / ``pre_period_weights=`` select a window), the treated
group's counterfactual survival is imputed from the control group's cumulative
hazard and the treated baseline (Theorem 1), and the **absorption ATT**
``E[Y_t - Y_t(0) | treated]`` is reported at every post-treatment date
(positive = more cumulative exit than the counterfactual). The headline
``att`` is the uniform average over the post-treatment dates.

Inference is the paper's whole-individual pooled bootstrap (Appendix B,
Algorithm 1): each draw resamples complete histories, recomputes everything,
and the reported pointwise intervals and simultaneous (max-|t|) band are
centered absolute-deviation bands. The Algorithm 2 fixed-anchor
pre-treatment specification test is reported separately in
``results.pretest``. Every inference family is either fully available or
fully withheld with a named ``inference_status``; failed draws are never
retried or silently dropped.

.. note::

   Requirements: exactly one row per (individual, date) on a common, equally
   spaced numeric time grid; a fixed 0/1 group indicator; a 0/1 absorbing
   outcome (baseline absorption is allowed and those individuals stay in the
   estimand). ``last_pre_period`` (the last untreated date) is required and
   never inferred. Covariates, staggered adoption, censoring, survey weights
   and cluster dependence are not supported in this version; the CD model
   can extrapolate an invalid counterfactual curve (survival above one or a
   decreasing cumulative hazard), which is reported through
   ``curve_status`` / ``period_status`` with the post-period inference
   withheld. The warning names the remedy that applies: a violation at a
   fitted pre-treatment date cannot be repaired by a shorter horizon (change
   ``method`` or the fitting window); a violation at a later post date admits
   an explicit refit on the dates before the first invalid post date; a
   violation at the first post date admits no valid shorter horizon. See
   ``docs/methodology/REGISTRY.md`` for the full contract.

**When to use DurationDiD:**

- A binary, absorbing outcome observed over time for a treated and a
  control group, with a common intervention date
- You want the effect on the probability of having exited by each
  post-treatment date, identified through hazards rather than outcome levels
- Individuals are independent draws (the bootstrap resamples individuals)

**Reference:** Deaner, B., & Ku, H. (2026). Causal Duration Analysis with
Diff-in-Diff. arXiv:2405.05220v2.
https://arxiv.org/abs/2405.05220v2

.. module:: diff_diff.duration_did

DurationDiD
-----------

Main estimator class.

.. autoclass:: diff_diff.DurationDiD
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

   .. rubric:: Methods

   .. autosummary::

      ~DurationDiD.fit
      ~DurationDiD.get_params
      ~DurationDiD.set_params

DurationDiDResults
------------------

Results container: headline inference, per-date effects with pointwise and
simultaneous bands, survival curves, fitted coefficient, bootstrap
diagnostics, and the pretest.

.. autoclass:: diff_diff.duration_did_results.DurationDiDResults
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Methods

   .. autosummary::

      ~DurationDiDResults.summary
      ~DurationDiDResults.to_dict
      ~DurationDiDResults.to_dataframe
      ~DurationDiDResults.aggregate

DurationDiDPretestResults
-------------------------

The Algorithm 2 fixed-anchor pre-treatment specification test.

.. autoclass:: diff_diff.duration_did_results.DurationDiDPretestResults
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Methods

   .. autosummary::

      ~DurationDiDPretestResults.summary
      ~DurationDiDPretestResults.to_dataframe
      ~DurationDiDPretestResults.to_dict

Example Usage
-------------

Every block below builds its own absorbing panel from population survival
curves (one uniform draw per individual), so it runs on its own.

Basic CD fit with bootstrap inference:

.. code-block:: python

    import numpy as np
    import pandas as pd
    from diff_diff import DurationDiD

    rng = np.random.default_rng(0)
    n, T = 500, 8
    control = 0.8 * np.exp(-np.cumsum(np.r_[0.0, 0.15 + 0.02 * np.arange(2, T + 1)]))
    treated = 0.6 * np.exp(-np.cumsum(np.r_[0.0, 0.20 + 0.02 * np.arange(2, T + 1)
                                              + 0.3 * (np.arange(2, T + 1) > 4)]))
    group = np.repeat([1, 0], n)
    curves = np.where(group[:, None] == 1, treated[None, :], control[None, :])
    exited = (rng.uniform(size=2 * n)[:, None] > curves).astype(int)
    data = pd.DataFrame({
        "unit": np.repeat(np.arange(2 * n), T),
        "time": np.tile(np.arange(1, T + 1), 2 * n),
        "treated": np.repeat(group, T),
        "exited": exited.ravel(),
    })

    results = DurationDiD(method="cd", n_bootstrap=200, seed=42).fit(
        data, outcome="exited", unit="unit", time="time", treatment="treated",
        last_pre_period=4,
    )
    print(results.summary())
    print(results.to_dataframe())          # per-date ATT, CI, simultaneous band
    print(results.pretest.summary())       # Algorithm 2 specification test

Proportional hazards with a fitting window on the last two pre-treatment
dates and explicit weights:

.. code-block:: python

    import numpy as np
    import pandas as pd
    from diff_diff import DurationDiD

    rng = np.random.default_rng(1)
    n, T = 400, 8
    control = 0.8 * np.exp(-np.cumsum(np.r_[0.0, 0.15 + 0.02 * np.arange(2, T + 1)]))
    treated = 0.6 * np.exp(-np.cumsum(np.r_[0.0, 1.5 * (0.15 + 0.02 * np.arange(2, T + 1))
                                              + 0.3 * (np.arange(2, T + 1) > 4)]))
    group = np.repeat([1, 0], n)
    curves = np.where(group[:, None] == 1, treated[None, :], control[None, :])
    data = pd.DataFrame({
        "unit": np.repeat(np.arange(2 * n), T),
        "time": np.tile(np.arange(1, T + 1), 2 * n),
        "treated": np.repeat(group, T),
        "exited": (rng.uniform(size=2 * n)[:, None] > curves).astype(int).ravel(),
    })

    results = DurationDiD(method="ph", n_bootstrap=100, seed=7).fit(
        data, outcome="exited", unit="unit", time="time", treatment="treated",
        last_pre_period=4, pre_periods=[3, 4], pre_period_weights=[1, 3],
    )
    print(results.coefficient)                  # fitted hazard ratio
    print(results.pre_periods, results.pre_period_weights)
    es = results.aggregate("event_study")       # unified event-study container
    print(es.to_dataframe())

Comparison with related estimators
----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Estimator
     - Identifying assumption
     - Outcome / design
   * - ``DurationDiD``
     - Constant gap (CD) or ratio (PH) between the groups' *untreated hazards*
     - Binary absorbing outcome, two groups, common timing
   * - ``DifferenceInDifferences`` / ``TwoWayFixedEffects``
     - Parallel trends in outcome *levels*
     - Any outcome; on an absorbing indicator this forces diverging hazards
   * - ``ChangesInChanges``
     - Distributional (monotone outcome model)
     - Continuous outcomes, 2x2
   * - ``CallawaySantAnna`` and the staggered family
     - Parallel trends, possibly conditional, per cohort
     - Staggered adoption; not a duration model
