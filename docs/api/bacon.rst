Bacon Decomposition (Goodman-Bacon 2021)
=========================================

Diagnostic decomposition of Two-Way Fixed Effects (TWFE) estimators for
staggered treatment designs.

This module implements the Goodman-Bacon (2021) decomposition, which reveals
that a TWFE estimate with variation in treatment timing is a weighted average
of all possible 2x2 Difference-in-Differences comparisons. The decomposition
exposes the implicit comparisons that drive the TWFE estimate -- including
potentially problematic "forbidden comparisons" where already-treated units
serve as controls -- and quantifies their relative importance.

**When to use BaconDecomposition:**

- You have a staggered adoption design and want to diagnose whether the TWFE
  estimate is driven by clean or problematic comparisons
- You need to assess the severity of heterogeneous treatment effect bias in
  existing TWFE results
- You want to understand *why* TWFE and robust estimators (e.g.,
  Callaway-Sant'Anna) produce different estimates
- You are deciding whether a simple TWFE specification is adequate or whether
  a robust staggered estimator is needed

**Reference:** Goodman-Bacon, A. (2021). Difference-in-differences with
variation in treatment timing. *Journal of Econometrics*, 225(2), 254-277.

.. module:: diff_diff.bacon

BaconDecomposition
------------------

Main estimator class for the Goodman-Bacon decomposition.

.. autoclass:: diff_diff.BaconDecomposition
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

   .. rubric:: Methods

   .. autosummary::

      ~BaconDecomposition.fit
      ~BaconDecomposition.get_params
      ~BaconDecomposition.set_params

BaconDecompositionResults
-------------------------

Results container for the Bacon decomposition.

.. autoclass:: diff_diff.bacon.BaconDecompositionResults
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Methods

   .. autosummary::

      ~BaconDecompositionResults.summary
      ~BaconDecompositionResults.print_summary
      ~BaconDecompositionResults.to_dataframe
      ~BaconDecompositionResults.weight_by_type
      ~BaconDecompositionResults.effect_by_type

Comparison2x2
-------------

Container for an individual 2x2 DiD comparison within the decomposition.

.. autoclass:: diff_diff.bacon.Comparison2x2
   :members:
   :undoc-members:
   :show-inheritance:

Convenience Function
--------------------

.. autofunction:: diff_diff.bacon_decompose

Example Usage
-------------

Basic usage::

    from diff_diff import BaconDecomposition, generate_staggered_data

    data = generate_staggered_data(n_units=200, n_periods=12,
                                    cohort_periods=[4, 6, 8], seed=42)

    bacon = BaconDecomposition()
    results = bacon.fit(data, outcome='outcome', unit='unit',
                        time='period', first_treat='first_treat')
    results.print_summary()

Visualizing with ``plot_bacon``::

    from diff_diff import plot_bacon

    # Scatter plot of 2x2 estimates vs weights, colored by comparison type
    ax = plot_bacon(results)
    ax.figure.show()

Interpreting the decomposition::

    # Convert to DataFrame for detailed inspection
    df = results.to_dataframe()
    print(df[['treated_group', 'control_group', 'comparison_type',
              'estimate', 'weight']])

    # Check weight breakdown by comparison type
    weights = results.weight_by_type()
    print(f"Treated vs Never-treated: {weights['treated_vs_never']:.1%}")
    print(f"Earlier vs Later:         {weights['earlier_vs_later']:.1%}")
    print(f"Later vs Earlier:         {weights['later_vs_earlier']:.1%}")

    # Compare weighted average effects across comparison types
    effects = results.effect_by_type()
    for comp_type, effect in effects.items():
        if effect is not None:
            print(f"  {comp_type}: {effect:.4f}")

Using exact weights for publication-quality results::

    bacon = BaconDecomposition(weights='exact')
    results = bacon.fit(data, outcome='outcome', unit='unit',
                        time='period', first_treat='first_treat')

    # Verify the weighted sum closely matches the TWFE estimate
    print(f"TWFE estimate:       {results.twfe_estimate:.4f}")
    print(f"Decomposition error: {results.decomposition_error:.6f}")

When Is TWFE Reliable?
----------------------

The Bacon decomposition helps answer whether a standard TWFE regression is
adequate for a particular dataset. As a rule of thumb:

- **TWFE is likely reliable** when the weight on "later vs earlier" (forbidden)
  comparisons is small, or when 2x2 estimates are similar across all comparison
  types. This suggests treatment effect heterogeneity is not meaningfully
  biasing the TWFE estimate.

- **TWFE may be unreliable** when forbidden comparisons carry substantial weight
  *and* their estimates differ markedly from the clean comparisons. In this
  case, consider using a robust staggered estimator such as
  :class:`~diff_diff.CallawaySantAnna`, :class:`~diff_diff.SunAbraham`, or
  :class:`~diff_diff.StackedDiD`.
