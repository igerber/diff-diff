.. meta::
   :description: Hands-on diff-diff tutorials — 33 Jupyter notebooks covering basic 2x2 DiD, staggered adoption, synthetic DiD, power analysis, and business applications.
   :keywords: DiD tutorial, difference-in-differences examples, causal inference notebooks

Tutorials
=========

Every tutorial is a runnable Jupyter notebook. If you are new to diff-diff,
start with :doc:`Basic DiD <01_basic_did>`; if you are measuring a business
intervention, start with the Business Applications track.

Business Applications
---------------------

Practitioner walkthroughs built around marketing and policy scenarios.

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: Brand Awareness Surveys
      :link: 17_brand_awareness_survey
      :link-type: doc

      Measure campaign impact on brand awareness with complex survey data
      and staggered rollouts.

   .. grid-item-card:: Geo-Experiments
      :link: 18_geo_experiments
      :link-type: doc

      Measure lift from a campaign that ran in a subset of geographic
      markets using Synthetic DiD.

   .. grid-item-card:: dCDH Marketing Pulse
      :link: 19_dcdh_marketing_pulse
      :link-type: doc

      Analyze on/off pulse campaigns with the de Chaisemartin-D'Haultfoeuille
      estimator for reversible treatments.

   .. grid-item-card:: HAD Brand Campaign
      :link: 20_had_brand_campaign
      :link-type: doc

      Estimate per-dollar lift when every market receives a different spend
      intensity.

   .. grid-item-card:: HAD Pre-test Workflow
      :link: 21_had_pretest_workflow
      :link-type: doc

      Run the pre-test diagnostics on the brand-campaign panel before
      trusting the estimate.

   .. grid-item-card:: Survey-Weighted HAD
      :link: 22_had_survey_design
      :link-type: doc

      Heterogeneous-adoption DiD on a BRFSS-shape survey rollout with design
      weights.

   .. grid-item-card:: Spillover DiD (TVA)
      :link: 23_spillover_tva
      :link-type: doc

      Handle treatment that spills over onto nearby control units in a
      TVA-style worked example.

   .. grid-item-card:: Composition Drift & Calibration
      :link: 26_composition_drift_calibration
      :link-type: doc

      Survey calibration for causal DiD when who answers changes over time.

   .. grid-item-card:: Distributional Effects (CiC)
      :link: 27_cic_distributional_effects
      :link-type: doc

      See which quantiles moved when the average hides the action, with
      Changes-in-Changes.

   .. grid-item-card:: MMM Calibration: PyMC-Marketing
      :link: 29_mmm_calibration_pymc
      :link-type: doc

      Turn a geo spend-boost experiment into a PyMC-Marketing lift test and
      watch the MMM's ROI posterior correct - executed end-to-end.

   .. grid-item-card:: MMM Calibration: Meridian
      :link: 30_mmm_calibration_meridian
      :link-type: doc

      Export a geo-holdout launch as Meridian's ROI prior and calibration
      mask, then fit the real model - executed end-to-end.

.. toctree::
   :maxdepth: 1
   :caption: Business Applications
   :hidden:

   Brand Awareness Surveys <17_brand_awareness_survey>
   Geo-Experiments <18_geo_experiments>
   dCDH Marketing Pulse <19_dcdh_marketing_pulse>
   HAD Brand Campaign <20_had_brand_campaign>
   HAD Pre-test Workflow <21_had_pretest_workflow>
   Survey-Weighted HAD <22_had_survey_design>
   Spillover DiD (TVA) <23_spillover_tva>
   Composition Drift & Calibration <26_composition_drift_calibration>
   Distributional Effects (CiC) <27_cic_distributional_effects>
   MMM Calibration: PyMC-Marketing <29_mmm_calibration_pymc>
   MMM Calibration: Meridian <30_mmm_calibration_meridian>

Fundamentals
------------

The core DiD toolkit, from your first 2x2 design to real published datasets.

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: Basic DiD
      :link: 01_basic_did
      :link-type: doc

      Your first 2x2 DiD: column-name and formula interfaces, covariates,
      fixed effects, and robust inference.

   .. grid-item-card:: Staggered DiD
      :link: 02_staggered_did
      :link-type: doc

      Handle staggered treatment adoption with modern heterogeneity-robust
      estimators.

   .. grid-item-card:: Synthetic DiD
      :link: 03_synthetic_did
      :link-type: doc

      Combine DiD with synthetic-control weighting (Arkhangelsky et al.
      2021).

   .. grid-item-card:: Triple Difference (DDD)
      :link: 08_triple_diff
      :link-type: doc

      Add a second comparison dimension when treatment requires satisfying
      two criteria.

   .. grid-item-card:: Real-World Examples
      :link: 09_real_world_examples
      :link-type: doc

      Classic datasets end to end: Card & Krueger minimum wage, Castle
      Doctrine, and more.

.. toctree::
   :maxdepth: 1
   :caption: Fundamentals
   :hidden:

   Basic DiD <01_basic_did>
   Staggered DiD <02_staggered_did>
   Synthetic DiD <03_synthetic_did>
   Triple Difference (DDD) <08_triple_diff>
   Real-World Examples <09_real_world_examples>

Advanced Methods
----------------

Modern estimators for designs the basic toolkit cannot handle.

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: TROP
      :link: 10_trop
      :link-type: doc

      Triply robust panel estimator with factor-model adjustment (Athey et
      al. 2025).

   .. grid-item-card:: Imputation DiD
      :link: 11_imputation_did
      :link-type: doc

      The efficient imputation estimator of Borusyak, Jaravel & Spiess
      (2024).

   .. grid-item-card:: Two-Stage DiD
      :link: 12_two_stage_did
      :link-type: doc

      Gardner (2022) two-stage residualized estimation for staggered
      designs.

   .. grid-item-card:: Stacked DiD
      :link: 13_stacked_did
      :link-type: doc

      Stacked event studies with corrective weights (Wing, Freedman &
      Hollingsworth 2024).

   .. grid-item-card:: Continuous DiD
      :link: 14_continuous_did
      :link-type: doc

      Dose-response treatment effects with continuous treatment intensity.

   .. grid-item-card:: Efficient DiD
      :link: 15_efficient_did
      :link-type: doc

      Semiparametrically efficient ATT estimation (Chen, Sant'Anna & Xie
      2025).

   .. grid-item-card:: Survey-Aware DiD
      :link: 16_survey_did
      :link-type: doc

      DiD on stratified survey microdata with design-based inference.

   .. grid-item-card:: Wooldridge ETWFE
      :link: 16_wooldridge_etwfe
      :link-type: doc

      Wooldridge's extended two-way fixed effects via pooled OLS.

   .. grid-item-card:: Synthetic Control for Policy
      :link: 25_synthetic_control_policy
      :link-type: doc

      Single-treated-unit policy evaluation with two routes to inference.

   .. grid-item-card:: Regression Discontinuity
      :link: 28_rdd_scholarship_illusion
      :link-type: doc

      Sharp and fuzzy RD from plot to estimate, when a naive cutoff
      comparison overstates the effect fivefold.

   .. grid-item-card:: LWDiD Rolling Transformations
      :link: 31_lwdid
      :link-type: doc

      Turn the panel into a cross-section, get exact small-sample
      inference, and replicate the Prop 99 and Walmart applications.

   .. grid-item-card:: Double ML DiD (Chang 2020)
      :link: 32_dml_did
      :link-type: doc

      Cross-fitted ML nuisances for nonlinear confounding - watch a
      misspecified linear model fail where the sieve recovers the truth.


.. toctree::
   :maxdepth: 1
   :caption: Advanced Methods
   :hidden:

   TROP <10_trop>
   Imputation DiD <11_imputation_did>
   Two-Stage DiD <12_two_stage_did>
   Stacked DiD <13_stacked_did>
   Continuous DiD <14_continuous_did>
   Efficient DiD <15_efficient_did>
   Survey-Aware DiD <16_survey_did>
   Wooldridge ETWFE <16_wooldridge_etwfe>
   Synthetic Control for Policy <25_synthetic_control_policy>
   Regression Discontinuity (RDD) <28_rdd_scholarship_illusion>
   LWDiD Rolling Transformations <31_lwdid>
   Double ML DiD (Chang 2020) <32_dml_did>

Study Design
------------

Assess identifying assumptions and size your study before committing to it.

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: Parallel Trends Diagnostics
      :link: 04_parallel_trends
      :link-type: doc

      Assess pre-treatment evidence relevant to parallel trends, check
      test power, and run the full diagnostic suite.

   .. grid-item-card:: Honest DiD Sensitivity
      :link: 05_honest_did
      :link-type: doc

      Robust inference under parallel-trends violations (Rambachan & Roth
      2023).

   .. grid-item-card:: Power Analysis
      :link: 06_power_analysis
      :link-type: doc

      Compute minimum detectable effects and sample sizes before running
      the study.

   .. grid-item-card:: Pre-Trends Power
      :link: 07_pretrends_power
      :link-type: doc

      Check whether your pre-trends test can actually detect violations
      (Roth 2022).

   .. grid-item-card:: Staggered vs Collapsed Power
      :link: 24_staggered_vs_collapsed_power
      :link-type: doc

      Staggered rollout or a simple 2x2? A power-analysis decision guide.

.. toctree::
   :maxdepth: 1
   :caption: Study Design
   :hidden:

   Parallel Trends Diagnostics <04_parallel_trends>
   Honest DiD Sensitivity <05_honest_did>
   Power Analysis <06_power_analysis>
   Pre-Trends Power <07_pretrends_power>
   Staggered vs Collapsed Power <24_staggered_vs_collapsed_power>
