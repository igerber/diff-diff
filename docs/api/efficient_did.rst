Efficient Difference-in-Differences
====================================

Semiparametrically efficient ATT estimator for staggered adoption designs
from Chen, Sant'Anna & Xie (2025).

This module implements the efficiency-bound-attaining estimator that:

1. **Achieves the semiparametric efficiency bound** for ATT(g,t) estimation on the no-covariate path
2. **Optimally weights** across comparison groups and baselines via the
   inverse covariance matrix Ω*
3. **Supports two PT assumptions**: PT-All (overidentified, tighter SEs) and
   PT-Post (just-identified, matches CS for post-treatment effects)
4. **Uses EIF-based inference** for analytical standard errors and multiplier
   bootstrap

.. note::

   EfficientDiD supports a doubly-robust covariate path with all nuisances
   estimated nonparametrically: sieve-based propensity score ratios and a
   sieve outcome regression (polynomial basis, AIC/BIC order selection),
   plus the kernel-smoothed conditional covariance. The DR property ensures
   consistency if either the outcome regression or the PS ratio is correctly
   specified, and because the nuisances are growing sieves/kernels the
   covariate path attains the semiparametric efficiency bound asymptotically
   under the paper's regularity conditions (degree 1 reproduces a linear
   working model, and ``sieve_k_max=1`` forces all covariate-path sieves to
   degree 1). Pass column names to the ``covariates``
   parameter on ``fit()``. See ``docs/methodology/REGISTRY.md`` for the full
   contract.

**When to use EfficientDiD:**

- Staggered adoption design where you want **maximum efficiency** on the no-covariate path
- You believe parallel trends holds across all pre-treatment periods (PT-All)
- You want tighter confidence intervals than Callaway-Sant'Anna
- You need a formal efficiency benchmark for comparing estimators

For covariate-adjusted designs, the doubly-robust path is consistent under
either outcome-regression or propensity-ratio correctness and attains the
efficiency bound under the paper's regularity conditions, with all nuisances
(sieve propensity ratio, sieve outcome regression, kernel covariance)
estimated nonparametrically.

**Reference:** Chen, X., Sant'Anna, P. H. C., & Xie, H. (2025). Efficient
Difference-in-Differences and Event Study Estimators.

.. module:: diff_diff.efficient_did

EfficientDiD
-------------

Main estimator class for Efficient Difference-in-Differences.

.. autoclass:: diff_diff.EfficientDiD
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

   .. rubric:: Methods

   .. autosummary::

      ~EfficientDiD.fit
      ~EfficientDiD.get_params
      ~EfficientDiD.set_params

EfficientDiDResults
-------------------

Results container for Efficient DiD estimation.

.. autoclass:: diff_diff.efficient_did_results.EfficientDiDResults
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Methods

   .. autosummary::

      ~EfficientDiDResults.aggregate
      ~EfficientDiDResults.summary
      ~EfficientDiDResults.print_summary
      ~EfficientDiDResults.to_dataframe

   .. rubric:: Attributes

   .. autosummary::

      ~EfficientDiDResults.reference_period

EDiDBootstrapResults
--------------------

Bootstrap inference results for Efficient DiD.

.. autoclass:: diff_diff.efficient_did_bootstrap.EDiDBootstrapResults
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

Example Usage
-------------

Basic usage::

    from diff_diff import EfficientDiD, generate_staggered_data

    data = generate_staggered_data(n_units=300, n_periods=10,
                                    cohort_periods=[4, 6, 8], seed=42)

    edid = EfficientDiD(pt_assumption="all")
    results = edid.fit(data, outcome='outcome', unit='unit',
                       time='period', first_treat='first_treat')
    results.print_summary()

    # Aggregate post-fit - recomputed from retained EIFs, no refit:
    print(results.aggregate('event_study').to_dataframe())
    print(results.aggregate('group').to_dataframe())
    print(results.aggregate('total').to_dataframe())  # estimator-owned total (3.10)

PT-Post mode (matches CS for post-treatment ATT)::

    edid_post = EfficientDiD(pt_assumption="post")
    results_post = edid_post.fit(data, outcome='outcome', unit='unit',
                                  time='period', first_treat='first_treat')
    print(f"PT-All ATT:  {results.overall_att:.4f} (SE={results.overall_se:.4f})")
    print(f"PT-Post ATT: {results_post.overall_att:.4f} (SE={results_post.overall_se:.4f})")

Bootstrap inference::

    edid_boot = EfficientDiD(pt_assumption="all", n_bootstrap=999, seed=42)
    results_boot = edid_boot.fit(data, outcome='outcome', unit='unit',
                                  time='period', first_treat='first_treat')
    print(f"Bootstrap SE: {results_boot.overall_se:.4f}")
    print(f"Bootstrap CI: [{results_boot.overall_conf_int[0]:.4f}, "
          f"{results_boot.overall_conf_int[1]:.4f}]")

Comparison with Other Staggered Estimators
------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 27 27 26

   * - Feature
     - EfficientDiD
     - CallawaySantAnna
     - ImputationDiD
   * - Approach
     - Optimal EIF-based weighting
     - Separate 2x2 DiD aggregation
     - Impute Y(0) via FE model
   * - PT assumption
     - PT-All (stronger) or PT-Post
     - Conditional PT
     - Strict exogeneity
   * - Efficiency
     - Achieves the semiparametric bound on the no-covariate path; the doubly-robust covariate path attains it too (under regularity conditions) via sieve nuisances
     - Not efficient
     - Efficient under homogeneity
   * - Covariates
     - Supported (doubly robust, sieve-based PS ratio + sieve outcome regression)
     - Supported (OR, IPW, DR)
     - Supported
   * - Bootstrap
     - Multiplier bootstrap (EIF)
     - Multiplier bootstrap
     - Multiplier bootstrap
   * - PT-Post equivalence
     - Matches CS post-treatment ATT(g,t)
     - Baseline
     - Different framework
