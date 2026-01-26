Triply Robust Panel (TROP)
==========================

Triply Robust Panel estimator for panel data with factor confounding.

This module implements the methodology from Athey, Imbens, Qu & Viviano (2025),
which combines three robustness components:

1. **Nuclear norm regularized factor model**: Estimates interactive fixed effects
   via matrix completion with nuclear norm penalty ||L||_*

2. **Exponential distance-based unit weights**: ω_j = exp(-λ_unit × d(j,i))
   where d(j,i) is the pairwise RMSE between units over pre-treatment periods

3. **Exponential time decay weights**: θ_s = exp(-λ_time × |t-s|)
   weighting periods by proximity to the specific treatment period t

**When to use TROP:**

- Suspected **factor structure** in the data (e.g., economic cycles, regional shocks)
- **Unobserved time-varying confounders** that affect units differently over time
- Standard parallel trends may be violated due to latent common factors
- Reasonably long pre-treatment period to estimate factors

**Reference:** Athey, S., Imbens, G. W., Qu, Z., & Viviano, D. (2025). Triply Robust
Panel Estimators. *Working Paper*. `arXiv:2508.21536 <https://arxiv.org/abs/2508.21536>`_

.. module:: diff_diff.trop

TROP
----

Main estimator class for Triply Robust Panel estimation.

.. autoclass:: diff_diff.TROP
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

   .. rubric:: Methods

   .. autosummary::

      ~TROP.fit
      ~TROP.get_params
      ~TROP.set_params

TROPResults
-----------

Results container for TROP estimation.

.. autoclass:: diff_diff.trop.TROPResults
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Methods

   .. autosummary::

      ~TROPResults.summary
      ~TROPResults.print_summary
      ~TROPResults.to_dict
      ~TROPResults.to_dataframe
      ~TROPResults.get_treatment_effects_df
      ~TROPResults.get_unit_effects_df
      ~TROPResults.get_time_effects_df

Convenience Function
--------------------

.. autofunction:: diff_diff.trop

Tuning Parameters
-----------------

TROP uses leave-one-out cross-validation (LOOCV) to select three tuning parameters:

.. list-table::
   :header-rows: 1
   :widths: 15 35 50

   * - Parameter
     - Description
     - Effect
   * - ``λ_time``
     - Time weight decay
     - Higher values weight periods closer to treatment more heavily
   * - ``λ_unit``
     - Unit distance decay
     - Higher values weight similar control units more heavily
   * - ``λ_nn``
     - Nuclear norm penalty
     - Higher values encourage lower-rank factor structure

Estimation Methods
------------------

TROP supports two estimation methods via the ``method`` parameter:

**Two-Step Method** (``method='twostep'``, default)

The default method follows Algorithm 2 from the paper:

1. **Grid search with LOOCV**: For each (λ_time, λ_unit, λ_nn) combination,
   compute cross-validation score by treating control observations as pseudo-treated

2. **Per-observation estimation**: For each treated observation (i, t):

   a. Compute observation-specific weights θ^{i,t} and ω^{i,t}
   b. Fit weighted model: Y = α + β + L + ε with nuclear norm penalty on L
   c. Compute τ̂_{it} = Y_{it} - α̂_i - β̂_t - L̂_{it}

3. **Average**: ATT = mean(τ̂_{it}) over all treated observations

This provides the **triple robustness** property (Theorem 5.1):
the estimator is consistent if any one of the three components
(unit weights, time weights, factor model) is correctly specified.

**Joint Method** (``method='joint'``)

An alternative approach that estimates a single scalar treatment effect:

1. **Compute weights**: Distance-based unit and time weights computed once
   (distance to center of treated block, RMSE to average treated trajectory)

2. **Joint optimization**: Solve weighted least squares problem

   .. math::

      \min_{\mu, \alpha, \beta, L, \tau} \sum_{i,t} \delta_{it} (Y_{it} - \mu - \alpha_i - \beta_t - L_{it} - W_{it} \tau)^2 + \lambda_{nn} \|L\|_*

   where τ is a **single scalar** (homogeneous treatment effect).

3. **With low-rank** (finite λ_nn): Uses alternating minimization between
   weighted LS for (μ, α, β, τ) and soft-threshold SVD for L.

The joint method is **faster** (single optimization vs N_treated optimizations)
but assumes **homogeneous treatment effects** across all treated observations.

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Feature
     - Two-Step (default)
     - Joint
   * - Treatment effect
     - Per-observation τ_{it}
     - Single scalar τ
   * - Flexibility
     - Heterogeneous effects
     - Homogeneous assumption
   * - Speed
     - Slower (N_treated fits)
     - Faster (single fit)
   * - Weights
     - Observation-specific
     - Global (center of treated block)

Use ``method='twostep'`` when treatment effects may vary across observations.
Use ``method='joint'`` for faster estimation when effects are expected to be homogeneous.

Example Usage
-------------

Basic usage::

    from diff_diff import TROP

    trop = TROP(
        lambda_time_grid=[0.0, 0.5, 1.0, 2.0],
        lambda_unit_grid=[0.0, 0.5, 1.0, 2.0],
        lambda_nn_grid=[0.0, 0.1, 1.0],
        n_bootstrap=200,
        seed=42
    )

    # Note: TROP infers treatment periods from the treatment indicator column.
    # The treatment column should be an absorbing state (D=1 for all periods
    # during and after treatment starts).
    results = trop.fit(
        data,
        outcome='y',
        treatment='treated',
        unit='unit_id',
        time='period'
    )
    results.print_summary()

Quick estimation with convenience function::

    from diff_diff import trop

    results = trop(
        data,
        outcome='y',
        treatment='treated',
        unit='unit_id',
        time='period',
        n_bootstrap=200
    )

Using the joint method for faster estimation::

    from diff_diff import TROP

    # Joint method: single scalar treatment effect via weighted LS
    trop_joint = TROP(
        method='joint',  # Use joint weighted least squares
        lambda_time_grid=[0.0, 0.5, 1.0, 2.0],
        lambda_unit_grid=[0.0, 0.5, 1.0, 2.0],
        lambda_nn_grid=[0.0, 0.1, 1.0],
        n_bootstrap=200,
        seed=42
    )
    results_joint = trop_joint.fit(data, outcome='y', treatment='treated',
                                    unit='unit_id', time='period')

    # Compare methods
    trop_twostep = TROP(method='twostep', ...)  # Default
    results_twostep = trop_twostep.fit(data, ...)
    print(f"Two-step ATT: {results_twostep.att:.3f}")
    print(f"Joint ATT: {results_joint.att:.3f}")

Examining factor structure::

    # Get the estimated factor matrix
    L = results.factor_matrix
    print(f"Effective rank: {results.effective_rank:.2f}")

    # Individual treatment effects
    effects_df = results.get_treatment_effects_df()
    print(effects_df)

Comparison with Synthetic DiD
-----------------------------

TROP extends Synthetic DiD by adding factor model adjustment:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Feature
     - Synthetic DiD
     - TROP
   * - Unit weights
     - Constrained to sum to 1
     - Exponential distance-based
   * - Time weights
     - Constrained to sum to 1
     - Exponential time decay
   * - Factor adjustment
     - None
     - Nuclear norm regularized L
   * - Robustness
     - Doubly robust
     - Triply robust

Use **SDID** when parallel trends is plausible. Use **TROP** when you suspect
factor confounding (regional shocks, economic cycles, latent factors).
