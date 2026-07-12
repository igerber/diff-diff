Changes-in-Changes (CiC) and Quantile DiD
=========================================

Distributional difference-in-differences for the canonical 2x2 design (two
groups, two periods) with continuous outcomes, from Athey & Imbens (2006).
Where standard DiD delivers a mean effect, these estimators recover the entire
counterfactual outcome distribution of the treated group, yielding quantile
treatment effects on the treated alongside the ATT.

``ChangesInChanges`` (alias ``CiC``) builds the counterfactual distribution
``F_10(F_00^{-1}(F_01(y)))``: each treated pre-period outcome is ranked in the
control pre-period distribution and pushed through the control post-period
quantile function. The model is invariant to monotone transformations of the
outcome (levels vs logs give consistent answers) and places no testable
restrictions on continuous data.

``QDiD`` is the quantile-by-quantile DiD comparison estimator the same paper
formalizes: it adds the control group's over-time quantile change to the
treated pre-period quantile. Athey & Imbens recommend CiC over QDiD - QDiD's
justifying model is not scale-invariant and imposes testable restrictions (a
warning fires when they appear violated).

.. note::

   Point estimation matches the R ``qte`` package (v1.3.1) exactly - CiC via
   R type-1 (ceiling-order-statistic) quantiles, the paper's empirical-inverse
   convention, and QDiD via qte's additive type-7 form (see the labeled Note
   in ``docs/methodology/REGISTRY.md`` on its finite-sample deviation from the
   paper's transformation form). Inference is bootstrap-only in this release
   (``n_bootstrap=200`` default, seedable): panel mode resamples units with
   both periods together, repeated cross-section mode draws a pooled row
   resample; SEs are replicate SDs with symmetric normal-approximation
   intervals, plus qte's sup-t critical value for uniform bands at a fixed
   95% level (independent of ``alpha``). Covariates, discrete-outcome bounds,
   analytical standard errors, and staggered designs are deferred - see
   ``docs/methodology/REGISTRY.md`` for the documented scope.

**When to use ChangesInChanges:**

- Exactly two groups and two periods, continuous outcome, and you care about
  effect heterogeneity across the outcome distribution (which quantiles moved,
  not just the mean)
- You want results invariant to monotone rescaling of the outcome
- CiC quantile effects are point-identified on the interior range where the
  treated pre-period distribution overlaps the control pre-period support;
  effects outside it keep their point estimates but report NaN inference

**Reference:** Athey, S., & Imbens, G. W. (2006). Identification and Inference
in Nonlinear Difference-in-Differences Models. *Econometrica*, 74(2), 431-497.

.. module:: diff_diff.changes_in_changes

ChangesInChanges
----------------

Main changes-in-changes estimator class.

.. autoclass:: diff_diff.ChangesInChanges
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Methods

   .. autosummary::

      ~ChangesInChanges.fit
      ~ChangesInChanges.get_params
      ~ChangesInChanges.set_params

QDiD
----

Quantile difference-in-differences comparison estimator.

.. autoclass:: diff_diff.QDiD
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Methods

   .. autosummary::

      ~QDiD.fit
      ~QDiD.get_params
      ~QDiD.set_params

ChangesInChangesResults
-----------------------

Results container shared by both estimators (the ``estimator`` field records
which produced it; ``QDiDResults`` is an alias of this class).

.. autoclass:: diff_diff.changes_in_changes_results.ChangesInChangesResults
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Methods

   .. autosummary::

      ~ChangesInChangesResults.summary
      ~ChangesInChangesResults.print_summary
      ~ChangesInChangesResults.to_dataframe
      ~ChangesInChangesResults.to_dict
      ~ChangesInChangesResults.uniform_bands

Example Usage
-------------

Estimate quantile treatment effects in a 2x2 design::

    import numpy as np
    import pandas as pd
    from diff_diff import ChangesInChanges

    rng = np.random.default_rng(0)
    n = 400
    treated = np.repeat([1, 0], n // 2)
    u = rng.normal(0, 1, n)
    y_pre = u + rng.normal(0, 0.3, n)
    y_post = u + 0.5 + rng.normal(0, 0.3, n) + treated * (0.8 + 0.4 * (u > 0))
    data = pd.DataFrame({
        "unit": np.tile(np.arange(n), 2),
        "post": np.repeat([0, 1], n),
        "treated": np.tile(treated, 2),
        "y": np.concatenate([y_pre, y_post]),
    })

    cic = ChangesInChanges(n_bootstrap=200, seed=42)
    results = cic.fit(data, outcome="y", treatment="treated", time="post")
    results.print_summary()

    # Quantile effects table and simultaneous bands
    qte = results.to_dataframe("quantiles")
    bands = results.uniform_bands()

Panel mode (same units in both periods) changes only the bootstrap::

    cic_panel = ChangesInChanges(n_bootstrap=200, seed=42, panel=True)
    results_panel = cic_panel.fit(
        data, outcome="y", treatment="treated", time="post", unit="unit"
    )

QDiD as a comparison estimator::

    from diff_diff import QDiD

    qdid = QDiD(n_bootstrap=200, seed=42)
    results_qdid = qdid.fit(data, outcome="y", treatment="treated", time="post")

Comparison with related estimators
----------------------------------

.. list-table::
   :header-rows: 1

   * - Estimator
     - Design
     - Output
     - Key assumption
   * - ``DifferenceInDifferences``
     - 2x2
     - Mean ATT
     - Parallel trends (additive, scale-dependent)
   * - ``ChangesInChanges``
     - 2x2
     - ATT + quantile effects
     - ``h(u, t)`` monotone in scalar unobservable; ``U`` time-invariant within groups
   * - ``QDiD``
     - 2x2
     - ATT + quantile effects
     - Additive quantile model (scale-dependent, testable restrictions)
   * - ``CallawaySantAnna``
     - Staggered
     - Mean ATT(g, t)
     - Parallel trends conditional on covariates
