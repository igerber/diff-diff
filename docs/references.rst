References
==========

This library implements methods from the following scholarly works.

Difference-in-Differences
-------------------------

- **Ashenfelter, O., & Card, D. (1985).** "Using the Longitudinal Structure of Earnings to Estimate the Effect of Training Programs." *The Review of Economics and Statistics*, 67(4), 648-660. https://doi.org/10.2307/1924810

- **Card, D., & Krueger, A. B. (1994).** "Minimum Wages and Employment: A Case Study of the Fast-Food Industry in New Jersey and Pennsylvania." *The American Economic Review*, 84(4), 772-793. https://www.jstor.org/stable/2118030

- **Angrist, J. D., & Pischke, J.-S. (2009).** *Mostly Harmless Econometrics: An Empiricist's Companion*. Princeton University Press. Chapter 5: Differences-in-Differences.

Two-Way Fixed Effects
---------------------

- **Wooldridge, J. M. (2010).** *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). MIT Press.

- **Imai, K., & Kim, I. S. (2021).** "On the Use of Two-Way Fixed Effects Regression Models for Causal Inference with Panel Data." *Political Analysis*, 29(3), 405-415. https://doi.org/10.1017/pan.2020.33

Wooldridge ETWFE
----------------

- **Wooldridge, J. M. (2025).** "Two-Way Fixed Effects, the Two-Way Mundlak Regression, and Difference-in-Differences Estimators." *Empirical Economics*, 69(5), 2545-2587. (Published version of NBER Working Paper 29154.)

  Primary source for the saturated OLS ETWFE design implemented in our ``WooldridgeDiD`` class.

- **Wooldridge, J. M. (2023).** "Simple Approaches to Nonlinear Difference-in-Differences with Panel Data." *The Econometrics Journal*, 26(3), C31-C66. https://doi.org/10.1093/ectj/utad016

  Secondary source for the logit/Poisson QMLE (ASF-based ATT) extensions in ``WooldridgeDiD``.

Robust Standard Errors
----------------------

- **White, H. (1980).** "A Heteroskedasticity-Consistent Covariance Matrix Estimator and a Direct Test for Heteroskedasticity." *Econometrica*, 48(4), 817-838. https://doi.org/10.2307/1912934

- **MacKinnon, J. G., & White, H. (1985).** "Some Heteroskedasticity-Consistent Covariance Matrix Estimators with Improved Finite Sample Properties." *Journal of Econometrics*, 29(3), 305-325. https://doi.org/10.1016/0304-4076(85)90158-7

- **Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011).** "Robust Inference With Multiway Clustering." *Journal of Business & Economic Statistics*, 29(2), 238-249. https://doi.org/10.1198/jbes.2010.07136

Wild Cluster Bootstrap
----------------------

- **Wu, C. F. J. (1986).** "Jackknife, Bootstrap and Other Resampling Methods in Regression Analysis." *The Annals of Statistics*, 14(4), 1261-1295. https://doi.org/10.1214/aos/1176350142

  Source of the ``sqrt(n_h/(n_h-1))`` Bessel small-sample correction applied to within-stratum bootstrap multipliers in :func:`diff_diff.bootstrap_utils.apply_stratum_centering`.

- **Liu, R. Y. (1988).** "Bootstrap Procedures Under Some Non-I.I.D. Models." *The Annals of Statistics*, 16(4), 1696-1708. https://doi.org/10.1214/aos/1176351062

- **Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2008).** "Bootstrap-Based Improvements for Inference with Clustered Errors." *The Review of Economics and Statistics*, 90(3), 414-427. https://doi.org/10.1162/rest.90.3.414

- **Davidson, R., & Flachaire, E. (2008).** "The Wild Bootstrap, Tamed at Last." *Journal of Econometrics*, 146(1), 162-169. https://doi.org/10.1016/j.jeconom.2008.08.003

  Wild bootstrap for heteroskedastic regression. Cited as an ingredient — the within-cluster mean-zero requirement on the multipliers (the canonical ``η_g`` centering that makes the wild bootstrap recover heteroskedasticity-robust inference) is the analog applied within each stratum in :func:`diff_diff.bootstrap_utils.apply_stratum_centering`. The paper does NOT cover stratified-survey designs directly; the stratification extension is the library synthesis (see REGISTRY § "Note (Stute stratified survey-bootstrap calibration)").

- **Kreiss, J.-P., & Lahiri, S. N. (2012).** "Bootstrap Methods for Time Series." In *Time Series Analysis: Methods and Applications* (Vol. 30, pp. 3-26). Elsevier. https://doi.org/10.1016/B978-0-444-53858-1.00001-6

  Bootstrap methods for time series. Cited as a methodological touchstone for the family of block-bootstrap / within-block centering arguments that underpins the wild-cluster-bootstrap extension to stratified PSU sampling. The exact composition (within-stratum demean + ``sqrt(n_h/(n_h-1))`` Bessel rescale on PSU multipliers applied before the per-obs broadcast in a wild-residual refit-in-loop bootstrap for a nonlinear empirical-process functional like the Stute CvM) is NOT in any single paper — it is a library synthesis of these ingredients.

- **Webb, M. D. (2014).** "Reworking Wild Bootstrap Based Inference for Clustered Errors." Queen's Economics Department Working Paper No. 1315. https://www.econ.queensu.ca/sites/econ.queensu.ca/files/qed_wp_1315.pdf

- **MacKinnon, J. G., & Webb, M. D. (2018).** "The Wild Bootstrap for Few (Treated) Clusters." *The Econometrics Journal*, 21(2), 114-135. https://doi.org/10.1111/ectj.12107

- **Djogbenou, A. A., MacKinnon, J. G., & Nielsen, M. Ø. (2019).** "Asymptotic Theory and Wild Bootstrap Inference with Clustered Errors." *Journal of Econometrics*, 212(2), 393-412. https://doi.org/10.1016/j.jeconom.2019.04.035

  Theorem 2 establishes empirical-process consistency of the cluster wild bootstrap for nonlinear smooth-functional consumers of per-obs residuals — the theoretical anchor for the stratified Stute CvM survey-bootstrap in :func:`diff_diff.had_pretests.stute_test` / :func:`stute_joint_pretest` (Phase 4.5 C strata extension).

- **Hlávka, Z., & Hušková, M. (2020).** "Multivariate Tests of Independence and the Wild Bootstrap." *Computational Statistics & Data Analysis*, 152, 107048. https://doi.org/10.1016/j.csda.2020.107048

  Vector-valued wild bootstrap consistency for empirical-process functionals (§3 condition on shared per-replicate multipliers preserving cross-component dependence). The theoretical anchor for the multi-horizon joint Stute survey-bootstrap in :func:`diff_diff.had_pretests.stute_joint_pretest` (the same ``psu_mults[b, :]`` row is shared across horizons within each replicate, preserving cross-horizon empirical-process dependence).

Nonparametric Bias-Corrected Inference
--------------------------------------

- **Calonico, S., Cattaneo, M. D., & Titiunik, R. (2014).** "Robust Nonparametric Confidence Intervals for Regression-Discontinuity Designs." *Econometrica*, 82(6), 2295-2326. https://doi.org/10.3982/ECTA11757

  Source of the bias-combined design matrix used by the in-house ``lprobust`` port that backs ``HeterogeneousAdoptionDiD`` Phase 1c (continuous-dose paths) for the bias-corrected weighted-robust SE.

- **Calonico, S., Cattaneo, M. D., & Titiunik, R. (2015).** "Optimal Data-Driven Regression Discontinuity Plots." *Journal of the American Statistical Association*, 110(512), 1753-1769. https://doi.org/10.1080/01621459.2015.1017578

  Source of the :class:`diff_diff.RDPlot` diagnostic: IMSE-optimal and mimicking-variance bin-count selectors for evenly- and quantile-spaced RD plots (spacings and polynomial-regression implementations), and the WIMSE implied-weights map behind the reported scale factors.

- **Calonico, S., Cattaneo, M. D., & Farrell, M. H. (2018).** "On the Effect of Bias Estimation on Coverage Accuracy in Nonparametric Inference." *Journal of the American Statistical Association*, 113(522), 767-779. https://doi.org/10.1080/01621459.2017.1285776

- **Calonico, S., Cattaneo, M. D., & Farrell, M. H. (2019).** "nprobust: Nonparametric Kernel-Based Estimation and Robust Bias-Corrected Inference." *Journal of Statistical Software*, 91(8), 1-33. https://doi.org/10.18637/jss.v091.i08

  CCF (2018, 2019) is the underlying ``nprobust`` machinery (MSE-optimal bandwidth selection and robust bias-corrected CIs) that ``HeterogeneousAdoptionDiD`` ports in-house for the continuous-dose paths.

- **Calonico, S., Cattaneo, M. D., Farrell, M. H., & Titiunik, R. (2017).** "rdrobust: Software for regression-discontinuity designs." *The Stata Journal*, 17(2), 372-404. https://doi.org/10.1177/1536867X1701700208

  The rdrobust software reference that :class:`diff_diff.RegressionDiscontinuity` parity-targets (CRAN 4.0.0): the 10-selector bandwidth menu, mass-point handling, and the three-row conventional / bias-corrected / robust output.

- **Calonico, S., Cattaneo, M. D., Farrell, M. H., & Titiunik, R. (2019).** "Regression Discontinuity Designs Using Covariates." *The Review of Economics and Statistics*, 101(3), 442-451. https://doi.org/10.1162/rest_a_00760

  Covariate-adjusted RD (additive common-coefficient specification and its consistency conditions) - the methodology behind :class:`diff_diff.RegressionDiscontinuity`'s ``fit(..., covariates=[...])`` adjustment (same estimand, covariate-aware bandwidths, balance as the operative testable condition).

- **Feir, D., Lemieux, T., & Marmer, V. (2016).** "Weak Identification in Fuzzy Regression Discontinuity Designs." *Journal of Business & Economic Statistics*, 34(2), 185-196. https://doi.org/10.1080/07350015.2015.1024836

  The weak-identification analysis behind :class:`diff_diff.RegressionDiscontinuity`'s fuzzy weak-first-stage warning (CCT 2014 cites the working-paper version); FLM's weak-IV-robust confidence sets are a documented follow-up seam.

Survey-Design Inference (Taylor-Series Linearization)
-----------------------------------------------------

- **Binder, D. A. (1983).** "On the Variances of Asymptotically Normal Estimators from Complex Surveys." *International Statistical Review*, 51(3), 279-292. https://doi.org/10.2307/1402588

  Foundational TSL (Taylor-Series Linearization) variance derivation used across diff-diff's survey-aware estimators (``compute_survey_if_variance`` and the per-estimator influence-function compositions, including the dCDH and HeterogeneousAdoptionDiD ``survey_design=`` paths).

- **Gerber, I. (2026).** "Design-Based Variance Estimation for Modern Heterogeneity-Robust Difference-in-Differences Estimators." *arXiv:2605.04124* (stat.ME). https://arxiv.org/abs/2605.04124

  Proposition 1 shows that the influence-function representations of 15 modern DiD estimators (including TwoStageDiD, explicitly derived in the Appendix) satisfy Binder's (1983) smoothness conditions, so standard stratified-cluster Taylor linearization produces design-consistent SEs. SpilloverDiD's Wave E.1 survey-design integration composes this result with the Wave D Gardner GMM first-stage uncertainty correction; see ``docs/methodology/REGISTRY.md`` SpilloverDiD section "Variance (Wave E.1)".

- **Lumley, T. (2004).** "Analysis of Complex Survey Samples." *Journal of Statistical Software*, 9(8), 1-19. https://doi.org/10.18637/jss.v009.i08

  Companion paper for the R ``survey`` package; defines the ``svydesign`` / ``svyglm`` / ``svrepdesign`` interface that diff-diff's survey path is cross-validated against (``tests/test_survey_r_crossvalidation.py``). The Lumley (2010) book below is the longer treatment.

- **Korn, E. L. & Graubard, B. I. (1990).** "Simultaneous Testing of Regression Coefficients with Complex Survey Data: Use of Bonferroni t Statistics." *The American Statistician*, 44(4), 270-276. https://doi.org/10.1080/00031305.1990.10475737

  Source for the survey degrees-of-freedom convention ``df = n_PSU - n_strata`` used for t-based inference on the TSL path (``ResolvedSurveyDesign.df_survey``; REGISTRY.md ``## Survey Data Support`` -> "Survey Degrees of Freedom").

- **Solon, G., Haider, S. J. & Wooldridge, J. M. (2015).** "What Are We Weighting For?" *Journal of Human Resources*, 50(2), 301-316. https://doi.org/10.3368/jhr.50.2.301

  The "when to weight" framework distinguishing precision, endogenous-sampling, and population-effect motivations for survey weights; cited in REGISTRY.md ``## Survey Data Support`` -> "Weighted Estimation".

- **Deville, J.-C. & Särndal, C.-E. (1992).** "Calibration Estimators in Survey Sampling." *Journal of the American Statistical Association*, 87(418), 376-382. https://doi.org/10.1080/01621459.1992.10475217

  The calibration/raking framework underlying post-stratified survey weights. diff-diff deliberately keeps calibration upstream (``SurveyDesign`` expects pre-calibrated weights); the ``docs/api/prep.rst`` "Weight calibration with balance" section and Tutorial 26 document the handoff.

- **Sarig, T., Galili, T. & Eilat, R. (2023).** "balance - a Python package for balancing biased data samples." *arXiv:2307.06024* (stat.CO). https://arxiv.org/abs/2307.06024

  Meta's ``balance`` package, the recommended upstream calibration companion. Its ``balance.interop.diff_diff`` adapter (balance >= 0.21) hands raked samples to diff-diff's survey-aware estimators; Tutorial 26 (``docs/tutorials/26_composition_drift_calibration.ipynb``) demonstrates the workflow, and ``tests/test_balance_interop_contract.py`` pins the consumed surface.

Placebo Tests and DiD Diagnostics
---------------------------------

- **Bertrand, M., Duflo, E., & Mullainathan, S. (2004).** "How Much Should We Trust Differences-in-Differences Estimates?" *The Quarterly Journal of Economics*, 119(1), 249-275. https://doi.org/10.1162/003355304772839588

- **Rosenbaum, P. R. (1996).** "Observational Studies and Nonrandomized Experiments." In *Handbook of Statistics*, Vol. 13, S. Ghosh & C. R. Rao, eds., 181-197. Elsevier. https://doi.org/10.1016/S0169-7161(96)13009-0

  Randomization-inference foundation for the ``PlaceboTests`` permutation test; cited by Bertrand-Duflo-Mullainathan (2004) footnote 11 as the basis for forming a randomization-inference test from the placebo-law distribution.

- **Phipson, B., & Smyth, G. K. (2010).** "Permutation P-values Should Never Be Zero: Calculating Exact P-values When Permutations Are Randomly Drawn." *Statistical Applications in Genetics and Molecular Biology*, 9(1), Article 39. https://doi.org/10.2202/1544-6115.1585

  Valid (slightly conservative) sampled randomization-inference p-value convention ``(1 + count) / (B + 1)`` used by ``permutation_test`` for randomly-drawn (with-replacement) permutations (the ``+1`` includes the observed statistic, giving an intrinsic ``1/(B+1)`` floor); the exact value is the full enumeration ``count/total``.

Synthetic Control Method
------------------------

- **Abadie, A., & Gardeazabal, J. (2003).** "The Economic Costs of Conflict: A Case Study of the Basque Country." *The American Economic Review*, 93(1), 113-132. https://doi.org/10.1257/000282803321455188

- **Abadie, A., Diamond, A., & Hainmueller, J. (2010).** "Synthetic Control Methods for Comparative Case Studies: Estimating the Effect of California's Tobacco Control Program." *Journal of the American Statistical Association*, 105(490), 493-505. https://doi.org/10.1198/jasa.2009.ap08746

- **Abadie, A., Diamond, A., & Hainmueller, J. (2015).** "Comparative Politics and the Synthetic Control Method." *American Journal of Political Science*, 59(2), 495-510. https://doi.org/10.1111/ajps.12116

- **Chernozhukov, V., Wüthrich, K., & Zhu, Y. (2021).** "An Exact and Robust Conformal Inference Method for Counterfactual and Synthetic Controls." *Journal of the American Statistical Association*, 116(536), 1849-1864. https://doi.org/10.1080/01621459.2021.1920957

- **Firpo, S., & Possebom, V. (2018).** "Synthetic Control Method: Inference, Sensitivity Analysis and Confidence Sets." *Journal of Causal Inference*, 6(2), 20160026. https://doi.org/10.1515/jci-2016-0026

Synthetic Difference-in-Differences
-----------------------------------

- **Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W., & Wager, S. (2021).** "Synthetic Difference-in-Differences." *American Economic Review*, 111(12), 4088-4118. https://doi.org/10.1257/aer.20190159

Triply Robust Panel (TROP)
--------------------------

- **Athey, S., Imbens, G. W., Qu, Z., & Viviano, D. (2025).** "Triply Robust Panel Estimators." *Working Paper*. https://arxiv.org/abs/2508.21536

  This paper introduces the TROP estimator which combines three robustness components:

  - **Factor model adjustment**: Low-rank factor structure via SVD removes unobserved confounders
  - **Unit weights**: Synthetic control style weighting for optimal comparison
  - **Time weights**: SDID style time weighting for informative pre-periods

  TROP is particularly useful when there are unobserved time-varying confounders with a factor structure that affect different units differently over time.

Triple Difference (DDD)
-----------------------

- **Ortiz-Villavicencio, M., & Sant'Anna, P. H. C. (2025).** "Better Understanding Triple Differences Estimators." *arXiv:2505.09942v3*. https://arxiv.org/abs/2505.09942v3

  This paper shows that common DDD implementations (taking the difference between two DiDs, or applying three-way fixed effects regressions) are generally invalid when identification requires conditioning on covariates. The ``TripleDifference`` class implements their regression adjustment, inverse probability weighting, and doubly robust estimators.

- **Gruber, J. (1994).** "The Incidence of Mandated Maternity Benefits." *American Economic Review*, 84(3), 622-641. https://www.jstor.org/stable/2118071

  Classic paper introducing the Triple Difference design for policy evaluation.

- **Olden, A., & Møen, J. (2022).** "The Triple Difference Estimator." *The Econometrics Journal*, 25(3), 531-553. https://doi.org/10.1093/ectj/utac010

Parallel Trends and Pre-Trend Testing
-------------------------------------

- **Roth, J. (2022).** "Pretest with Caution: Event-Study Estimates after Testing for Parallel Trends." *American Economic Review: Insights*, 4(3), 305-322. https://doi.org/10.1257/aeri.20210236

- **Lakens, D. (2017).** "Equivalence Tests: A Practical Primer for t Tests, Correlations, and Meta-Analyses." *Social Psychological and Personality Science*, 8(4), 355-362. https://doi.org/10.1177/1948550617697177

Honest DiD / Sensitivity Analysis
---------------------------------

The ``HonestDiD`` module implements sensitivity analysis methods for relaxing the parallel trends assumption.

- **Rambachan, A., & Roth, J. (2023).** "A More Credible Approach to Parallel Trends." *The Review of Economic Studies*, 90(5), 2555-2591. https://doi.org/10.1093/restud/rdad018

  This paper introduces the "Honest DiD" framework implemented in our ``HonestDiD`` class:

  - **Relative Magnitudes (ΔRM)**: Bounds post-treatment violations by a multiple of observed pre-treatment violations
  - **Smoothness (ΔSD)**: Bounds on second differences of trend violations, allowing for linear extrapolation of pre-trends
  - **Breakdown Analysis**: Finding the smallest violation magnitude that would overturn conclusions
  - **Robust Confidence Intervals**: Valid inference under partial identification

- **Roth, J., & Sant'Anna, P. H. C. (2023).** "When Is Parallel Trends Sensitive to Functional Form?" *Econometrica*, 91(2), 737-747. https://doi.org/10.3982/ECTA19402

  Discusses functional form sensitivity in parallel trends assumptions, relevant to understanding when smoothness restrictions are appropriate.

Multi-Period and Staggered Adoption
-----------------------------------

- **Borusyak, K., Jaravel, X., & Spiess, J. (2024).** "Revisiting Event-Study Designs: Robust and Efficient Estimation." *Review of Economic Studies*, 91(6), 3253-3285. https://doi.org/10.1093/restud/rdae007

  This paper introduces the imputation estimator implemented in our ``ImputationDiD`` class:

  - **Efficient imputation**: OLS on untreated observations, impute counterfactuals, aggregate
  - **Conservative variance**: Theorem 3 clustered variance estimator with auxiliary model
  - **Pre-trend test**: Independent of treatment effect estimation (Proposition 9)
  - **Efficiency gains**: ~50% shorter CIs than Callaway-Sant'Anna under homogeneous effects

- **Callaway, B., & Sant'Anna, P. H. C. (2021).** "Difference-in-Differences with Multiple Time Periods." *Journal of Econometrics*, 225(2), 200-230. https://doi.org/10.1016/j.jeconom.2020.12.001

- **Sant'Anna, P. H. C., & Zhao, J. (2020).** "Doubly Robust Difference-in-Differences Estimators." *Journal of Econometrics*, 219(1), 101-122. https://doi.org/10.1016/j.jeconom.2020.06.003

- **Sun, L., & Abraham, S. (2021).** "Estimating Dynamic Treatment Effects in Event Studies with Heterogeneous Treatment Effects." *Journal of Econometrics*, 225(2), 175-199. https://doi.org/10.1016/j.jeconom.2020.09.006

- **Gardner, J. (2022).** "Two-stage differences in differences." *arXiv preprint arXiv:2207.05943*. https://arxiv.org/abs/2207.05943

- **Butts, K., & Gardner, J. (2022).** "did2s: Two-Stage Difference-in-Differences." *The R Journal*, 14(1), 162-173. https://doi.org/10.32614/RJ-2022-048

- **Butts, K. (2023).** "Difference-in-Differences with Spatial Spillovers." *arXiv:2105.03737v3* (originally posted 2021). https://arxiv.org/abs/2105.03737

  Identifies the ring-indicator estimator implemented in our ``SpilloverDiD`` class. Section 2-3 covers non-staggered timing (Equations 5/6/8); Section 5 covers staggered timing via two-stage Gardner (Table 2). Section 3.1 (page 13) recommends Conley spatial-HAC for inference with cutoff = ``d_bar``. Section 4 Table 1 Panel A (TVA agriculture) is the empirical anchor for tutorial ``docs/tutorials/23_spillover_tva.ipynb`` (~40% understatement direction).

- **Kline, P., & Moretti, E. (2014).** "Local Economic Development, Agglomeration Economies, and the Big Push: 100 Years of Evidence from the Tennessee Valley Authority." *The Quarterly Journal of Economics*, 129(1), 275-331. https://doi.org/10.1093/qje/qjt034

  Original Tennessee Valley Authority application that Butts (2021) §4 revisits to illustrate spillover bias in the canonical place-based-intervention DiD. Cited as the motivating real-world example in ``docs/tutorials/23_spillover_tva.ipynb``.

- **Conley, T. G. (1999).** "GMM Estimation with Cross Sectional Dependence." *Journal of Econometrics*, 92(1), 1-45. https://doi.org/10.1016/S0304-4076(98)00084-0

  Primary source for the Conley spatial-HAC variance estimator. Equations 5-9 derive the spatial-kernel cross-product meat. Our ``diff_diff/conley.py`` implements the practitioner specializations (Bartlett / uniform kernels with haversine / euclidean metrics) cited in our ``SpilloverDiD`` Wave A/D Conley path and composed with Binder TSL + Gardner GMM in Wave E.2 (``_compute_stratified_conley_meat`` at ``diff_diff/two_stage.py``).

- **de Chaisemartin, C., & D'Haultfœuille, X. (2020).** "Two-Way Fixed Effects Estimators with Heterogeneous Treatment Effects." *American Economic Review*, 110(9), 2964-2996. https://doi.org/10.1257/aer.20181169

- **de Chaisemartin, C., & D'Haultfœuille, X. (2022, revised July 2023).** "Difference-in-Differences Estimators of Intertemporal Treatment Effects." *NBER Working Paper* 29873. https://www.nber.org/papers/w29873

  Dynamic companion to the 2020 paper. Web Appendix Section 3.7.3 contains the cohort-recentered plug-in variance formula implemented in our ``ChaisemartinDHaultfoeuille`` class.

- **Goodman-Bacon, A. (2021).** "Difference-in-Differences with Variation in Treatment Timing." *Journal of Econometrics*, 225(2), 254-277. https://doi.org/10.1016/j.jeconom.2021.03.014

- **Newey, W. K., & West, K. D. (1987).** "A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix." *Econometrica*, 55(3), 703-708. https://doi.org/10.2307/1913610

  Primary source for the Bartlett serial-HAC kernel weights `1 - |t-s|/(L+1)` used in time-series and panel HAC variance estimators. Our ``diff_diff/conley.py`` panel-block path at L949-965 hardcodes this kernel for the within-unit serial component (mirrors R ``conleyreg::time_dist``); SpilloverDiD's Wave E.2 follow-up composes the same kernel weights with Binder/Gerber TSL + Wave D Gardner GMM correction for the panel-block survey case (``_compute_stratified_serial_bartlett_meat`` at ``diff_diff/two_stage.py``; see REGISTRY section "Variance (Wave E.2 follow-up)").

- **Lumley, T. (2010).** *Complex Surveys: A Guide to Analysis Using R.* Hoboken, NJ: John Wiley & Sons. https://doi.org/10.1002/9780470580066

  Authoritative reference for the R ``survey`` package. §2.5 "Domains and subpopulations" documents the ``svyrecvar(subset(design, mask))`` zero-pad convention: subpopulation analyses preserve the full survey design (strata / PSU / FPC) and treat out-of-domain observations as zero-score padding rows rather than physically subsetting the design. SpilloverDiD's Wave E.3 adopts this convention at the ``_compute_gmm_corrected_meat`` boundary, matching the existing in-library precedents at ``diff_diff/imputation_aggregation.py::_compute_lead_coefficients`` (PreTrendsImputation lead regression) and ``diff_diff/prep.py:1401-1432`` (DCDH cell variance); see REGISTRY section "Variance (Wave E.3)".

- **Wing, C., Freedman, S. M., & Hollingsworth, A. (2024).** "Stacked Difference-in-Differences." *NBER Working Paper* 32054. https://www.nber.org/papers/w32054

- **Ustyuzhanin, V. (2026).** "Covariate-Balanced Weighted Stacked Difference-in-Differences." *arXiv preprint* arXiv:2604.02293v1. https://arxiv.org/abs/2604.02293

  Primary source for ``StackedDiD``'s covariate-balancing path (the ``balance="entropy"`` parameter). Adds a within-sub-experiment design stage — nonnegative control design weights ``b_{sa}`` — that composes with the Wing et al. (2024) corrective weights via the effective control mass into the final stacked weights ``W_{sa}``, so untreated counterfactual trends are estimated under *conditional* parallel trends while the treated-cohort weights (and hence the trimmed-aggregate-ATT estimand) are unchanged.

- **Hainmueller, J. (2012).** "Entropy Balancing for Causal Effects: A Multivariate Reweighting Method to Produce Balanced Samples in Observational Studies." *Political Analysis*, 20(1), 25-46. https://doi.org/10.1093/pan/mpr025

  Foundational source for entropy balancing. ``StackedDiD(balance="entropy")`` implements this within-sub-experiment reweighting: nonnegative control weights that exactly match the treated covariate means while minimizing the Kullback-Leibler divergence from uniform.

- **Chen, X., Sant'Anna, P. H. C., & Xie, H. (2025).** "Efficient Difference-in-Differences and Event Study Estimators." *arXiv preprint* arXiv:2506.17729v1. https://arxiv.org/abs/2506.17729v1 (also Cowles Foundation Discussion Paper No. 2470).

  Primary source for the optimal-weighting / PT-All / PT-Post efficient DiD implemented in our ``EfficientDiD`` class.

- **Baker, A., Callaway, B., Cunningham, S., Goodman-Bacon, A., & Sant'Anna, P. H. C. (2025).** "Difference-in-Differences Designs: A Practitioner's Guide." *arXiv preprint* arXiv:2503.13323. https://arxiv.org/abs/2503.13323

  Source for the 8-step practitioner workflow surfaced via ``diff_diff.get_llm_guide("practitioner")`` and the README ``## Practitioner Workflow`` section. See ``docs/methodology/REGISTRY.md`` for the diff-diff renumbering and per-step deviations.

Local Projections DiD
---------------------

- **Dube, A., Girardi, D., Jordà, Ò., & Taylor, A. M. (2025).** "A Local Projections Approach to Difference-in-Differences." *Journal of Applied Econometrics*, 40(7), 741-758. https://doi.org/10.1002/jae.70000 (Open Access; NBER Working Paper 31184; FRBSF Working Paper 2023-12.)

  Primary source for the ``LPDiD`` estimator: per-horizon long-difference local-projection regressions estimated on a "clean control" sample, yielding non-negatively-weighted event-study and pooled ATTs (variance-weighted by default; reweighted regression or regression adjustment for the equally-weighted ATT; premean-differenced base periods; covariates; non-absorbing extension). Reference Stata package ``lpdid`` (SSC s459273); R ``alexCardazzi/lpdid``. Paper review on file at ``docs/methodology/papers/dube-2025-review.md``.

- **Jordà, Ò. (2005).** "Estimation and Inference of Impulse Responses by Local Projections." *American Economic Review*, 95(1), 161-182. https://doi.org/10.1257/0002828053828518

  Origin of the local-projections method that LP-DiD adapts to the difference-in-differences setting.

Rolling-Transformation DiD (Lee-Wooldridge)
-------------------------------------------

- **Lee, S. J., & Wooldridge, J. M. (2025).** "A Simple Transformation Approach to Difference-in-Differences Estimation for Panel Data." SSRN Working Paper No. 4516518 (61-page revision, cover page June 8, 2026). https://ssrn.com/abstract=4516518 | https://doi.org/10.2139/ssrn.4516518

  Estimation core for the forthcoming ``LWDiD`` estimator: unit-specific rolling demeaning/detrending converts panel DiD into per-(cohort, period) cross-sectional treatment-effects problems (RA, IPW, doubly robust IPWRA, matching), with contributing-treated-unit weighted event-study aggregation (simplifying to cohort-size weights in balanced panels) and influence-function multiplier-bootstrap inference. Paper review on file at ``docs/methodology/papers/lee-wooldridge-2025-review.md``.

- **Lee, S. J., & Wooldridge, J. M. (2026).** "Simple Approaches to Inference with Difference-in-Differences Estimators with Small Cross-Sectional Sample Sizes." SSRN Working Paper No. 5325686 (36 pages, cover page February 3, 2026). https://ssrn.com/abstract=5325686 | https://doi.org/10.2139/ssrn.5325686

  Exact small-sample inference layer: collapsed cross-sectional regressions with exact ``t`` reference distributions (valid down to a single treated unit), HC3 and randomization-inference alternatives, and the composite-outcome aggregate regression for staggered designs. Reference Stata package ``lwdid`` (Hur, Lee & Wooldridge; SSC s459672), whose MIT-licensed ancillary datasets back ``load_prop99()`` and ``load_walmart()``. Paper review on file at ``docs/methodology/papers/lee-wooldridge-2026-review.md``.

Changes-in-Changes / Distributional DiD
---------------------------------------

- **Athey, S., & Imbens, G. W. (2006).** "Identification and Inference in Nonlinear Difference-in-Differences Models." *Econometrica*, 74(2), 431-497. https://doi.org/10.1111/j.1468-0262.2006.00668.x

  Primary source for the ``ChangesInChanges`` (alias ``CiC``) and ``QDiD`` estimators: nonlinear DiD recovering the treated group's full counterfactual outcome distribution and quantile treatment effects in the 2x2 design, with the quantile-DiD comparison estimator the paper formalizes alongside it. Point estimation matches the R ``qte`` package (v1.3.1, Callaway) exactly; bootstrap inference follows the same package's conventions. Paper review on file at ``docs/methodology/papers/athey-imbens-2006-review.md``; companion reviews: ``melly-santangelo-2015-review.md`` (covariates - implemented in qte's simplified form, see below), ``callaway-li-oka-2018-review.md`` (panel QTT, deferred), ``ciaccio-2024-review.md`` (staggered, deferred).

- **Melly, B., & Santangelo, G. (2015).** "The Changes-in-Changes Model with Covariates." *Working paper*, Bern University and European Commission - JRC. https://sites.google.com/site/blaisemelly/home/computer-programs/cic_stata

  Reference for conditional (covariate) changes-in-changes: per-cell quantile-regression conditional CDFs composed into a conditional CiC and integrated over the treated covariate distribution. ``ChangesInChanges``/``QDiD`` implement the simplified form of this pipeline that the R ``qte`` package ships as ``xformla`` (fixed 99-tau grid, per-observation imputation, treated pre-period integration); the paper's full estimator (monotonized CDFs, exchangeable bootstrap, uniform bands, specification test) is documented-deferred. The conditional-support diagnostic warning cites this paper's Assumption 4. Paper review on file at ``docs/methodology/papers/melly-santangelo-2015-review.md``.

- **Koenker, R., & Bassett, G. (1978).** "Regression Quantiles." *Econometrica*, 46(1), 33-50. https://doi.org/10.2307/1913643

  Source of the linear quantile-regression check-function objective that the covariate CiC/QDiD branch solves (as a linear program via ``scipy.optimize.linprog``/HiGHS, matching R ``quantreg::rq``'s Barrodale-Roberts solutions; see the solver Note in ``docs/methodology/REGISTRY.md``).

Continuous Treatment DiD
------------------------

- **Callaway, B., Goodman-Bacon, A., & Sant'Anna, P. H. C. (2024).** "Difference-in-Differences with a Continuous Treatment." *NBER Working Paper* 32117. https://www.nber.org/papers/w32117

  Primary source for ATT(d), ACRT, dose-response curves, and B-spline flexibility implemented in our ``ContinuousDiD`` class.

Heterogeneous Adoption (No-Untreated Designs)
---------------------------------------------

- **de Chaisemartin, C., Ciccia, D., D'Haultfœuille, X., & Knau, F. (2026).** "Difference-in-Differences Estimators When No Unit Remains Untreated." *arXiv preprint* arXiv:2405.04465v6. https://arxiv.org/abs/2405.04465

  Primary source for the Weighted Average Slope (WAS) estimator and its multi-period event-study extension implemented in our ``HeterogeneousAdoptionDiD`` class. Targets panels where no unit remains untreated at the post period and treatment dose ``D_{g,2}`` is nonnegative, using local-linear regression at the dose support boundary - both Design 1' (the QUG case with ``d̲ = 0``) and Design 1 (no QUG with ``d̲ > 0``) are supported.

Power Analysis
--------------

- **Bloom, H. S. (1995).** "Minimum Detectable Effects: A Simple Way to Report the Statistical Power of Experimental Designs." *Evaluation Review*, 19(5), 547-556. https://doi.org/10.1177/0193841X9501900504

- **Burlig, F., Preonas, L., & Woerman, M. (2020).** "Panel Data and Experimental Design." *Journal of Development Economics*, 144, 102458. https://doi.org/10.1016/j.jdeveco.2020.102458

  Essential reference for power analysis in panel DiD designs. Derives serial-correlation-robust (SCR) variance formulas (their Eq. 2) and shows that serial correlation can *raise or lower* the MDE depending on the cross-period covariance. The library's analytical panel-power path implements the **within-unit equicorrelated special case** of Eq. 2 — ``Var = σ²(1/n_T+1/n_C)(1/m+1/r)(1−ρ)`` (lineage: Frison & Pocock 1992 → McKenzie 2012 → Burlig et al.); the fully general SCR form with independent pre/post/cross-period covariances is not yet implemented. See ``docs/methodology/REGISTRY.md`` ``## PowerAnalysis``.

- **Djimeu, E. W., & Houndolo, D.-G. (2016).** "Power Calculation for Causal Inference in Social Science: Sample Size and Minimum Detectable Effect Determination." *Journal of Development Effectiveness*, 8(4), 508-527. https://doi.org/10.1080/19439342.2016.1244555

- **Frison, L., & Pocock, S. J. (1992).** "Repeated Measures in Clinical Trials: Analysis Using Mean Summary Statistics and Its Implications for Design." *Statistics in Medicine*, 11(13), 1685-1704. https://doi.org/10.1002/sim.4780111304

  Origin of the constant-correlation (equicorrelated) repeated-measures variance that Burlig, Preonas & Woerman (2020) generalize; the lineage source for the library's analytical panel-power formula.

- **McKenzie, D. (2012).** "Beyond Baseline and Follow-Up: The Case for More T in Experiments." *Journal of Development Economics*, 99(2), 210-221. https://doi.org/10.1016/j.jdeveco.2012.01.002

  Establishes the power gains from multiple pre/post periods in panel experiments (the ``(m+r)/(mr)`` period factor); the ψ = 0 special case of Burlig, Preonas & Woerman (2020) Eq. 2.

MMM Calibration Interop
-----------------------

- **PyMC Labs.** "Lift Test Calibration." *PyMC-Marketing documentation*. https://www.pymc-marketing.io/en/stable/notebooks/mmm/mmm_lift_test.html

  Defines the lift-test DataFrame schema (``channel``/dims/``x``/``delta_x``/``delta_y``/``sigma``) and the saturation-curve likelihood consumed by ``MMM.add_lift_test_measurements``, which ``to_pymc_marketing_lift_test`` targets.

- **Google.** "Set Custom Prior Distributions Using Past Experiments." *Google Meridian documentation*. https://developers.google.com/meridian/docs/advanced-modeling/set-custom-priors-past-experiments

  Documents the experiment-to-ROI-prior calibration workflow and caveats; the lognormal ``(mu, sigma)`` closed form emitted by ``to_meridian_roi_prior`` matches Meridian's ``prior_distribution.lognormal_dist_from_mean_std`` (verified against Meridian 1.7.0 source).

- **Google.** "Set the ROI Calibration Period." *Google Meridian documentation (configure-model guide)*. https://developers.google.com/meridian/docs/user-guide/configure-model

  Defines the ``roi_calibration_period`` contract ``meridian_calibration_mask`` builds against: an optional boolean array of shape ``(n_media_times, n_media_channels)``, True where a period participates in ROI calibration, time labels drawn from the model's ``time`` coordinate and channels ordered by ``data.media_channel``.

- **Zhou, G., Choe, Y., & Hetrakul, C. (2023).** "Calibrated MMM Better Predicts True ROAS." *Meta Marketing Science*. https://medium.com/@gufengzhou/calibrated-mmm-better-predicts-true-roas-d5adfc8abdc4

  Empirical motivation for experiment calibration: calibrating MMMs against lift experiments substantially reduces ROAS prediction error.

General Causal Inference
------------------------

- **Imbens, G. W., & Rubin, D. B. (2015).** *Causal Inference for Statistics, Social, and Biomedical Sciences: An Introduction*. Cambridge University Press.

- **Cunningham, S. (2021).** *Causal Inference: The Mixtape*. Yale University Press. https://mixtape.scunning.com/
