# Paper Review: Declaring and Diagnosing Research Designs

**Authors:** Graeme Blair, Jasper Cooper, Alexander Coppock, Macartan Humphreys ("Authors are listed in alphabetical order", p.838)
**Citation:** Blair, G., Cooper, J., Coppock, A., & Humphreys, M. (2019). Declaring and Diagnosing Research Designs. American Political Science Review, 113(3), 838-859.
**DOI:** https://doi.org/10.1017/S0003055419000194
**Source reviewed:** American Political Science Review 113(3), 22-page PDF = journal pp. 838-859 (main text pp. 838-856; references pp. 856-858; appendix pp. 858-859). Open access, CC BY 4.0. The PDF was reviewed externally and is not committed to the repository. Reproduce by downloading the article via the DOI above. All page citations below are JOURNAL page numbers. The online Supplementary Materials and the Dataverse replication archive that the paper points to were NOT reviewed. Transcription conventions: the paper's dashes are rendered as hyphens, typographic quotes as straight quotes, and Greek letters, mathematical symbols and diacritics in ASCII. Greek letters and mathematical symbols inside quoted passages are transliterated to ASCII (for example `tau`, `rho`, `+/-`, `x` for the multiplication sign).
**Review date:** 2026-11-22

> **Personal verification stamp (load-bearing items spot-checked against the source).** Beyond the multi-agent extraction, the reviewer read p.841 directly and verified Table 1 in full (all twelve diagnosands, their descriptions, and the M / I / D / A requirement marks). Also checked against the source: the "reports and reconciles" passage and the exploratory-versus-confirmatory sentence (p.852); the "nothing in the framework that enforces them" passage (p.856); the answer-strategy information sentence and the SUTVA qualification (p.858); the treatment-to-endline timing sentence (p.840); the QCA conditions and footnote 14 (p.848); and the cluster-level estimand and subgroup qualification in the replication example (pp.854-855). Every quoted passage of 40 or more characters was additionally run through a mechanical word-sequence comparison against the PDF text layer.

---

## Methodology Registry Entry

**Status: proposed text for a future REGISTRY section; no such section exists in `docs/methodology/REGISTRY.md` today; this file is a non-authoritative source audit.** The paper is the candidate source for the FRAMEWORK and VOCABULARY of a planned "design diagnosis" capability (declare a planned design, simulate or resample it before any real analysis, and report design properties per estimator). It is reviewed as a set with the companion review of Coglianese and Fang (2026). Nothing below is a methodology contract until a REGISTRY section is written and reviewed alongside an implementation.

*This is a FRAMEWORK paper, not an estimator paper: it proposes no estimator, states no theorem, and prints no numbered equation. The standard registry headings (estimator equation, standard errors) therefore do not apply and are replaced by the design framework, the diagnosand roster, the diagnosis procedure, and the design-comparison criteria. The registry-candidate text ends at the marker before `## Worked Examples and Results`; everything below that marker is audit material (worked examples, verbatim quotes, implementation notes, gaps) and is NOT part of the proposed registry text.*

## Design declaration and diagnosis framework (working title)

*The heading above is provisional. The paper's own terms are "design declaration", "design diagnosis", "diagnosand", and the acronym "MIDA"; the name of any library surface is to be decided in the implementation plan.*

**Primary source:** [Blair, G., Cooper, J., Coppock, A., & Humphreys, M. (2019). Declaring and Diagnosing Research Designs. American Political Science Review, 113(3), 838-859.](https://doi.org/10.1017/S0003055419000194)

**Key implementation requirements:**

*Scope of the source:*
- A general framework for "formally 'declaring' the analytically relevant features of a research design in a demonstrably complete manner, with applications to qualitative, quantitative, and mixed methods research" (abstract, p.838).
- Diagnosis is by simulation of the declared design: "Declaration of these features in code provides sufficient information for researchers and readers to use Monte Carlo techniques to diagnose properties such as power, bias, accuracy of qualitative causal inferences, and other 'diagnosands.'" (abstract, p.838)
- The framework is software-independent: "the principles of design declaration and diagnosis do not depend on any particular software implementation" (p.839).
- What the framework covers and omits: "MIDA captures the analysis-relevant features of a design, but it does not describe substantive elements, such as how theories are derived, how interventions are implemented, or even, qualitatively, how outcomes are measured." (p.840)
- The worked examples are experimental, survey, qualitative, matching, regression and regression-discontinuity designs. Difference-in-differences and panel designs are not among them - see "What the paper does NOT provide".

*Rationale (pp.838-839, 843-845):*
- "To select strong designs, we often rely on rules of thumb, simple power calculators, or principles from the methodological literature that typically address one component of a design while assuming optimal conditions for others. These relatively informal practices can result in the selection of suboptimal designs, or worse, designs that are simply too weak to deliver useful answers." (p.838)
- Four stated benefits: "It can facilitate the diagnosis of designs in terms of their ability to answer the questions we want answered under specified conditions; it can assist in the improvement of research designs through comparison with alternatives; it can enhance research transparency by making design choices explicit; and it can provide strategies to assist principled replication and reanalysis of published research." (p.839)
- On existing power tools: "researchers are generally not designing studies using the actual strategies that they will use to conduct analysis. From the perspective of the overall designs, the power calculations are providing the wrong answer." (p.843). "Power calculators do not generally focus on broader aspects of a design, like alternative assignment procedures or the choice of estimator." (pp.843-844). "Quantitative researchers tend to focus on power, when other diagnosands such as bias, coverage, or RMSE may also be important." (p.844)

*Design framework - the four elements (Section "Elements of a Research Design", pp.839-840):*

"At its most basic we think of a research design as including four elements `<M, I, D, A>`" (p.839). The acronym for the framework is "MIDA" (p.838). Definitions as printed:

| Symbol | Paper's term | Definition as printed | Contents |
|---|---|---|---|
| `M` | "causal model" (also "model of the world", "background model", pp.838-839) | "A causal **model**, M, of how the world works." (p.839) | Three core elements (pp.839-840): (1) "a specification of the variables X about which research is being conducted. This includes endogenous and exogenous variables (V and U respectively) and the ranges of these variables" - "sometimes called the *signature* of a model"; (2) "a specification of how each endogenous variable depends on other variables (the 'functional relations' or, as in Imbens and Rubin (2015), 'potential outcomes'), F"; (3) "a probability distribution over exogenous variables, P(U)" |
| `I` | "inquiry" | "An **inquiry**, I, about the distribution of variables, X, perhaps given interventions on some variables." (p.840) | "We let a^M denote the answer to I *under the model*. Conditional on the model, a^M is the value of the estimand, the quantity that the researcher wants to learn about." (p.840) |
| `D` | "data strategy" | "A **data** strategy, D, generates data d on X under model M with probability P_M(d\|D)." (p.840) | "The data strategy includes sampling strategies and assignment strategies, which we denote with P_S and P_Z respectively. Measurement techniques are also a part of data strategies and can be thought of as procedures by which unobserved latent variables are mapped (possibly with error) into observed variables." (p.840) |
| `A` | "answer strategy" | "An **answer** strategy, A, that generates answer a^A using data d." (p.840) | In the appendix: "functions that use information from realized data and the design, but do not have access to the full schedule of potential outcomes" (p.858) |

Notation as printed (p.840):

```
Inquiry examples:            Pr(X_1 | X_2 = 1)          conditional values of variables
                             Pr(X_1 | do(X_2 = 1))      values that would arise under interventions

a^M                          the answer to I under the model (the value of the estimand)
a^A                          the answer generated by A using data d
P_M(d | D)                   probability of data d under model M and data strategy D
P_S, P_Z                     sampling strategies, assignment strategies

Distribution of the answer:                      P_M(a^A | D)
Comparison with the correct answer under M:      P_M(a^A - a^M | D)
Under a different data strategy D':              P_M(a^A - a^M | D')
Under a different answer strategy A':            P_M(a^{A'} - a^M | D)
Under an alternative model M':                   P_M(a^A - a^{M'} | D)
```

- Central property: "A key feature of this bare specification is that if M, D, and A are sufficiently well described, the answer to question I has a distribution P_M(a^A|D). Moreover, one can construct a distribution of comparisons of this answer to the correct answer, under M" (p.840). Comparisons across alternative models are available "as long as these possess signatures that are consistent with inquiries and answer strategies" (p.840).
- A model is needed even for non-causal questions: "Though M is a causal model of the world, such a model can be used for both causal and non-causal questions of interest." (footnote 1, p.839)
- The inquiry is specified independently of the functional relations: in the paper's illustration, "The specification of this question depends on the signature of the model, but not the functional relations of the model. The answer provided by the model does of course depend on the functional relations." (p.840)
- Analytically relevant details that are not one of the four labels still enter: "if treatment effects decay, logistical details of data collection (such as the duration of time between a treatment being administered and endline data collection) may enter into the model. Similarly, if a researcher anticipates noncompliance, substantive knowledge of how treatments are taken up can be included in many parts of the design." (p.840)
- Estimand placement matters (appendix, p.859): "Order matters. Since we have defined the estimand before the sampling step, our estimand is the Population Average Treatment Effect, not the Sample Average Treatment Effect."

*Diagnostic statistics and diagnosands (Section "Diagnosands", p.840):*

The paper introduces "two further concepts, both functions of research designs. These are quantities that a researcher or a third party could calculate with respect to a design." (p.840)

> "A **diagnostic statistic** is a summary statistic generated from a 'run' of a design-that is, the results given a possible realization of variables, given the model and data strategy. For example the statistic: e = 'difference between the estimated and the actual average treatment effect' is a diagnostic statistic that requires specifying an estimand. The statistic s = 1(p <= 0.05), interpreted as 'the result is considered statistically significant at the 5% level,' is a diagnostic statistic that does not require specifying an estimand, but it does presuppose an answer strategy that reports a p-value." (p.840)

> "Diagnostic statistics are governed by probability distributions that arise because both the model and the data generation, given the model, may be stochastic." (p.840)

> "A **diagnosand** is a summary of the distribution of a diagnostic statistic. For example, (expected) *bias* in the estimated treatment effect is E(e) and statistical *power* is E(s)." (p.840)

The introduction glosses diagnosands as "quantitative summaries of design properties" (p.838). Formulas printed anywhere in the paper (all unnumbered):

```
e    = "difference between the estimated and the actual average treatment effect"      (p.840)
s    = 1(p <= 0.05)                                                                    (p.840)
bias  = E(e)                                                                           (p.840)
power = E(s)                                                                           (p.840)
bias  : "the distribution of bias given the model, a^A - a^M"                          (p.840)
bias  : E[a^A - a^M]            (printed as "E[a^A - a^M] != 0" for a biased design)   (p.849)
mean-squared-error = E[(a^A - a^M)^2]                                                  (p.849)
average answer = E[a^A]         ("the answer we expect the design to provide")         (p.854)
code:  bias = mean(estimate - estimand)                                 (Table 3, p.845; p.859)
```

No other diagnosand has a printed formula; the rest are defined in words only (Table 1 and the passages cited below).

*Diagnosand roster (Table 1, p.841: "Examples of Diagnosands and the Elements of the Model (M), Inquiry (I), Data Strategy (D), and Answer Strategy (A) Required in Order for a Design to be Diagnosand-Complete for Each Diagnosand"):*

The table is transcribed in full. A "Y" marks a check under "Required:"; a blank marks no check.

| Diagnosand | Description (verbatim) | M | I | D | A |
|---|---|---|---|---|---|
| Power | Probability of rejecting null hypothesis of no effect | Y | | Y | Y |
| Estimation bias | Expected difference between estimate and estimand | Y | Y | Y | Y |
| Sampling bias | Expected difference between population average treatment effect and sample average treatment effect (Imai, King, and Stuart 2008) | Y | Y | Y | |
| RMSE | Root mean-squared-error | Y | Y | Y | Y |
| Coverage | Probability the confidence interval contains the estimand | Y | Y | Y | Y |
| SD of estimates | Standard deviation of estimates | Y | | Y | Y |
| SD of estimands | Standard deviation of estimands | Y | Y | Y | |
| Imbalance | Expected distance of covariates across treatment conditions | Y | | Y | |
| Type S rate | Probability estimate has incorrect sign, if statistically significant (Gelman and Carlin 2014) | Y | Y | Y | Y |
| Exaggeration ratio | Expected ratio of absolute value of estimate to estimand, if statistically significant (Gelman and Carlin 2014) | Y | Y | Y | Y |
| Value for money | Probability that a decision based on estimated effect yields net benefits | Y | Y | Y | Y |
| Robustness | Joint probability of rejecting the null hypothesis across multiple tests | Y | | Y | Y |

Reading the `I` column:
- **Defined relative to an estimand (check under I):** estimation bias, sampling bias, RMSE, coverage, SD of estimands, Type S rate, exaggeration ratio, value for money.
- **Need only the estimates and tests, no estimand (no check under I):** power, SD of estimates, robustness.
- **Needs neither an inquiry nor an answer strategy:** imbalance (M and D only). Sampling bias and SD of estimands need no answer strategy.
- The paper states the same split in prose: "power-completeness requires that the answer strategy return a p-value and a significance threshold be specified. It does not, however, require a well-defined estimand, such as a true effect size (see Table 1 where, for a power diagnosand, there is no check under I). In contrast, bias- or RMSE-completeness does not require a hypothesis test, but does require the specification of an estimand." (p.842)

Diagnosands named in the text but not tabulated in Table 1:

| Diagnosand | Definition as printed | Page |
|---|---|---|
| Mean squared error | `E[(a^A - a^M)^2]`; also "mean squared error" in the discovery example | pp.849, 851, 855 |
| Average (expected) answer / mean estimate | "the average answer, i.e., the answer we expect the design to provide (E[a^A])"; Figure 2 legend "Mean estimate" | p.854 |
| Mean estimand | Figure 2 legend "Mean estimand"; "The dotted vertical line represents the true average effect (a^M)" | p.854 |
| Exaggeration ratio (second statement) | "the ratio of the absolute value of the estimate to the absolute value of the estimand, given that the estimated effect is significant at the alpha = 0.05 level. This diagnosand thus provides a measure of how much the design exaggerates effect sizes conditional on statistical significance." | p.855 |
| Out-of-sample bias | "an out-of-sample diagnosand, such as bias relative to the population average treatment effect" | p.838 |
| Probability of the right policy decision | "a diagnosand directly related to policy choices, such as the probability of making the right policy decision after research is conducted" | pp.838-839 |
| Posterior shift (Bayesian) | "the expected shift in the location and scale of the posterior distribution relative to the prior distribution"; results reported as "posterior standard deviation" and bias | p.846 |
| Correct recovery (qualitative) | algorithms "successfully recover the correct configuration 100% of the time" - conditional on "no error and the sample is large enough to ensure sufficient variation in the data" (p.848); abstract: "accuracy of qualitative causal inferences" | pp.838, 848 |
| Probability of producing a result | the split-sample strategy "is somewhat less likely to produce a result *at all*" | p.851 |
| Model-fit diagnosands | "for a range of values of X we could compare values of f_Y(X) to f*_Y(X), or employ familiar statistics of goodness of fit, such as the R^2" | p.851 |

Statements about the roster:
- Table 1 is a list of examples, not a closed set: "We list and describe some of these in Table 1" (p.841); "new diagnosands can be generated in response to substantive concerns" (p.856).
- Any user-supplied summary can be a diagnosand; the companion software "is able to calculate standard errors for any user-provided diagnosand" (footnote 11, p.845).
- The two informal entries of Table 1 are glossed in the text: "one might define an inference as 'robust' if the same inference is made under different analysis strategies. One might conclude that an intervention gives 'value for money' if estimates are of a certain size and be interested in the probability that a researcher in correct in concluding that an intervention provides value for money." (p.841). Reviewer note: this prose gloss of "robust" (same inference under different analysis strategies) is not the same wording as the Table 1 description (joint probability of rejecting across multiple tests); the paper does not reconcile the two.
- Qualitative designs: "We believe there is not yet a consensus around diagnosands for qualitative designs. However, in certain treatments clear analogues of diagnosands exist, such as sampling bias or estimation bias (e.g., Herron and Quinn 2016). There are indeed notions of power, coverage, and consistency for QCA researchers" (p.841)
- Bayesian: "analogous diagnosands can be used to assess Bayesian estimation strategies (see Rubin 1984), and as we illustrate below, some diagnosands are unique to Bayesian answer strategies." (p.841)
- Thresholds: "For some diagnosands, quality thresholds have been established through common practice, such as the standard power target of 0.80. Some researchers are unsatisfied unless the 'bias' diagnosand is exactly zero. Yet for most diagnosands, we only have a sense of better and worse, and improving one can mean hurting another, as in the classic bias-variance tradeoff. Our goal is not to dichotomize designs into high and low quality, but instead to facilitate the assessment of design quality on dimensions important to researchers." (pp.841-842)
- Standard error of the estimate is NOT named as a diagnosand anywhere. Standard errors appear as an output of the answer strategy (p.859), as the bootstrap standard error OF a diagnosand estimate (p.845), and in the replication discussion of covariate adjustment ("the standard errors are ever so slightly larger"; a regressor that could "increase standard error of the treatment effect estimate", p.855).

**Reviewer reconstruction (NOT printed in the paper, except the bias line).** With simulation draws `r = 1..m`, estimate `a^A_r`, estimand `a^M_r`, p-value `p_r`, confidence interval `[lo_r, hi_r]`, and threshold `alpha`, the Table 1 descriptions and the plug-in principle of footnote 12 are consistent with:

```
bias                = mean_r( a^A_r - a^M_r )                              (printed: mean(estimate - estimand))
RMSE                = sqrt( mean_r( (a^A_r - a^M_r)^2 ) )
power               = mean_r( 1(p_r <= alpha) )
coverage            = mean_r( 1(lo_r <= a^M_r <= hi_r) )
SD of estimates     = sd_r( a^A_r )
SD of estimands     = sd_r( a^M_r )
mean estimate       = mean_r( a^A_r )
Type S rate         = mean over { r : p_r <= alpha } of 1( sign(a^A_r) != sign(a^M_r) )
exaggeration ratio  = mean over { r : p_r <= alpha } of |a^A_r| / |a^M_r|
```

What the paper does fix: each run's diagnostic statistic uses THAT run's model-derived estimand (Table 3 computes `a^M` inside each simulation and then `estimate - estimand`, p.845), and a diagnosand summarizes the distribution of the per-run statistic (p.840); substituting an across-run average estimand is not presented as an alternative. The paper prints this per-run construction explicitly only for bias (`estimate - estimand`); applying it to coverage and the other estimand-relative diagnosands is a reviewer reconstruction consistent with it. The exaggeration ratio is an EXPECTED ratio, conditional on significance (Table 1, p.841; p.855), i.e. a mean of per-run ratios, not a ratio of means. Choices the paper leaves open (an implementation must make and document each): inclusive vs exclusive interval bounds for coverage; the standard-deviation denominator; and what the two conditional diagnosands return when NO simulated run is significant. (Which runs enter them is fixed: Type S rate and exaggeration ratio are defined "if statistically significant", Table 1, p.841, so only significant runs are in the conditioning set.)

*Diagnosand-completeness (Section "What is a Complete Research Design Declaration?", p.842):*
- Definition: "We propose a conditional conception of completeness: we say a design is 'diagnosand-complete' for a given diagnosand if that diagnosand can be calculated from the declared design. Thus a design that is diagnosand-complete for one diagnosand may not be for another." (p.842). Introduction: "We say a design declaration is 'diagnosand-complete' when a diagnosand can be estimated from the declaration. We do not have a general notion of a complete design, but rather adopt an approach in which the purposes of the design determine which diagnosands are valuable and in turn what features must be declared." (p.838)
- Power, as worded here: "Power is the probability of obtaining a statistically significant result. Equivalently, it is the probability that the p-value is lower than a critical value (e.g., 0.005)." (p.842). The value 0.005 is as printed; the diagnostic-statistic example on p.840 uses 0.05.
- Completeness is necessary, not sufficient: "Diagnosand-completeness is a desirable property to the extent that it means a diagnosand can be calculated. How useful diagnosand-completeness is depends on whether the diagnosand is worth knowing." (p.842)
- Completeness is conditional on the declared assumptions: "The utility of a diagnosis depends in part on whether the information underlying declaration is believable. For instance, a design may be bias-complete, but only under the assumptions of a given spillover structure. Readers might disagree with these assumptions. Even in this case, however, an advantage of declaration is a clarification of the conditions for completeness." (p.842)
- Community standards: "In practice, domain-specific standards might be agreed upon among members of particular research communities." (p.838)

*Diagnosis procedure (Section "Declaring and Diagnosing Research Designs in Practice", p.845; Table 3, p.845; appendix, pp.858-859):*

Diagnosis is carried out by Monte Carlo simulation of the declared design. Analytic diagnosis is not developed; it is mentioned only in passing ("Power calculations like these are done using formulae", p.843; "analytic results, when available, may extend over general classes of designs", p.856).

> "A design that can be declared in computer code can then be simulated in order to diagnose its properties. The approach to declaration that we advocate is one that conceives of a design as a concatenation of steps." (p.845)

> "A single simulation runs through these steps, calling each of the functions successively. A design diagnosis conducts m simulations, then summarizes the resulting distribution of diagnostic statistics in order to estimate the diagnosand." (p.845)

The simulation algorithm, transcribed from the middle and bottom panels of Table 3 (p.845):

```
Design simulation (1 draw)
  1  Draw a population u using P(U)                                     u   <- p_U()
  2  Generate potential outcomes using f_Y                              D   <- f_Y(u)
  3  Calculate estimand a^M                                             a_M <- I(D)
  4  Draw data, d, given model assumptions and data strategies          d   <- R(p_Z(p_S(D)))
  5  Calculate answers, a^A using A and d                               a_A <- A(d)
  6  Calculate a diagnostic statistic t using a^A and a^M               e   <- a_A["estimate"] - a_M["estimand"]

Design diagnosis (m draws)
  Declare a diagnosand         bias <- declare_diagnosands(bias = mean(estimate - estimand))
  Calculate a diagnosand       diagnose_design(design, diagnosands = bias, sims = m)
```

- **What is redrawn.** As printed, every draw starts from step 1: the population `u`, the potential outcomes, the estimand, the sample, the assignment, and the answer are all regenerated in each of the `m` simulations. The paper describes no variant that holds a population or a data set fixed across draws.
- **Declarations are processes, not realizations.** "Critically, the declaration is not a declaration of a particular realization of data but of a data generating *process*." (p.858). "Again sampling describes a sampling strategy and not an actual sample." (p.858)
- **Number of simulations.** No number is recommended. Table 3 writes `sims = m`; the appendix example uses `sims = 500` (p.859). "Diagnosands can be estimated with higher levels of precision by increasing m. However, simulations are often computationally expensive." (p.845)
- **Estimator of a diagnosand.** "In nearly all cases, diagnosands will be features of the distribution of a diagnostic statistic that, given i.i.d. sampling, can be consistently estimated via plug-in estimation (for example taking sample means). Our simulation procedure, by construction, yields i.i.d. draws of the diagnostic statistic." (footnote 12, p.846). The same footnote warns: "This procedure depends on the researcher choosing a 'good' diagnosand estimator."
- **Appendix restatement.** "Diagnosing the design involves simulating the design many times, then calculating the value of the diagnosand from the resulting simulations." (p.859). "The diagnosis returns an estimate of the diagnosands, along with other metadata associated with the simulations." (p.859). The appendix denotes "the diagnostic statistics, t(D, Y, f), and diagnosands, theta(D, Y, f, g)" and adds "For simplicity, the two can be coded as a single function." (p.859). The arguments of `t` and `theta` are not defined.

*Uncertainty in diagnosand estimates (p.845; footnotes 11-12):*

> "In order to assess whether researchers have conducted enough simulations to be confident in their diagnosand estimates, we recommend estimating the sampling distributions of the diagnosands via the nonparametric bootstrap. With the estimated diagnosand and its standard error, we can characterize our uncertainty about whether the range of likely values of the diagnosand compare favorably to reference values such as statistical power of 0.8." (pp.845-846)

- Footnote 11 (p.845): "In their paper on simulating clinical trials through Monte Carlo, Morris, White, and Crowther (2019) provide helpful analytic formula for deriving Monte Carlo standard errors for several diagnosands ('performance measures'). In the companion software, we adopt a non-parametric bootstrap approach that is able to calculate standard errors for any user-provided diagnosand."
- What is resampled, the number of bootstrap replicates, and the form of any interval are not stated. The only printed call sets `bootstrap_sims = FALSE` (p.859).

*Assumed model and effect sizes (pp.840-842, 846, 858):*
- A model must be assumed, and the paper says so directly: "Design diagnosis places a burden on researchers to come up with a causal model, M. Since researchers presumably want to learn about the model, declaring it in advance may seem to beg the question. Yet declaring a model is often unavoidable when diagnosing designs. In practice, doing so is already familiar to any researcher who has calculated the power of a design, which requires the specification of effect sizes." (p.846)
- Mitigation by sensitivity analysis: "The seeming arbitrariness of the declared model can be mitigated by assessing the sensitivity of diagnosis to alternative models and strategies, which is relatively straightforward given a diagnosand-complete design declaration." (p.846)
- Existing data may inform the model: "Further, researchers can inform their substantive models with existing data, such as baseline surveys." (p.846). Appendix: "Researchers will typically have a sense of the distribution of covariates from previous work, and may even have an existing data set of the units that will be in the study with background characteristics. Researchers should assess the sensitivity of their diagnosands to different assumptions about P(U)." (p.858). This is the full extent of the paper's discussion of existing or real data in a declaration; no procedure for it is given, and pilot data are not mentioned.
- Diagnosis is conditional on the model: "These design evaluations depend on the model, and so one might reasonably ask how performance would look were the model different" (p.841). Diagnosis "can help a researcher select from a range of designs, conditional upon beliefs about the world" (p.839). Diagnosands "can be estimated given conjectures about the world" (p.851).
- Two suggested ways to specify potential outcomes (appendix, p.858): "In many cases, the potential outcomes function (or its features) is the very thing that the study sets out to learn, so it can seem odd to assume features of it. We suggest two approaches to developing potential outcomes functions that will yield useful information about the quality of designs. First, consider a null potential outcomes function in which the variables of interest are set to have no effect on the outcome whatsoever. Diagnosands such as bias can then be assessed relative to a true estimand of zero. This approach will not work for diagnosands like power or the Type-S rate. Second, set a series of potential outcomes functions that correspond to competing theories. This approach enables the researcher to judge whether the design yields answers that help adjudicate between the theories."
- A grid of models is used in the worked examples: "we consider models M_1,..., M_K, where we let the interaction between treatments vary over the range -0.2 to +0.2" (p.850).
- Minimum detectable effects are mentioned once, as an analogy and not as a diagnosand: "Just as power calculators focus attention on minimum detectable effects, design declaration offers a tool to demonstrate design properties and how they change depending on researcher assumptions." (p.846)

*Redesign and design comparison (pp.840-841, 843, 850-852):*
- The paper has no section called "redesign"; its terms are "comparison with alternatives" (p.839) and "alternative designs" (p.841).
- General principle: "In all cases, the evaluation of a design depends on the assessment of a diagnosand, and comparing the diagnoses to what could be achieved under alternative designs." (p.841)
- Three axes of comparison, from the questions that open the Diagnosands section: "How good is the answer you expect to get from a given strategy? Would you do better, given some desideratum, with a different data strategy? With a different analysis strategy? How good is the strategy if the model is wrong in one way or another?" (p.840)
- Workflow: "Declaring the design's features in code enables direct exploration of alternative data and analysis strategies using simulated data; evaluating alternative strategies through diagnosis; and exploring the robustness of a chosen strategy to alternative models. Researchers can undertake each step before study implementation or data collection." (pp.851-852)
- Evaluation is multi-dimensional with no aggregate score: "Given that there are many possible diagnosands, the overall evaluation of a design is both multi-dimensional and qualitative." (p.841)
- Comparisons are design-specific: "This exercise should not be taken as proof of the superiority of one strategy over another in general; instead we learn about their relative performance for particular diagnosands for the specific design declared." (p.843)
- Power alone can mislead when comparing answer strategies: one estimator "is better powered than DIM and BFE because it produces biased variance estimates that lead to a coverage probability that is too low" (p.843).
- Sample size is varied as a redesign in one example only: the simulated sampling strategy is modified "to include 187 clusters instead of 50" and the exaggeration ratio compared (pp.854-855). No sample-size search procedure is described.
- A split of data into training and testing halves is itself diagnosed as an answer strategy: "With this design declared, one can assess what an optimal division of units into training and testing data might be given different hypothesized effect sizes." (p.851)

*Criteria for admitting alternative answer strategies (Section "Challenging Design Choices", pp.852-853; Table 4, p.853):*

Table 4 ("Diagnosis Results Given Alternative Assumptions About the Model and Alternative Answer Strategies") numbers four situations:

| | Author's assumed model | Alternative claims on model |
|---|---|---|
| Author's proposed answer strategy | 1 | 2 |
| Alternative answer strategy | 3 | 4 |

"In this context, there are several possible criteria for admitting alternative answer strategies" (p.853):

- "**Home Ground Dominance.** If ex ante the diagnostics for situation 3 are better than for 1 then this gives grounds to switch to 3. That is, if a critic can demonstrate that an alternative estimation strategy outperforms an original estimation strategy even under the data generating process assumed by an original researcher, then they have strong grounds to propose a change in strategies. Conversely, if an alternative estimation strategy produces different results, conditional on the data, but does not outperform the original strategy given the original assumptions, this gives grounds to question the reanalysis." (p.853)
- "**Robustness to Alternative Models.** If the diagnostics in situation 3 are as good as in 1 but are better in situation 4 than in situation 2 this provides a robustness argument for altering estimation strategies." (p.853)
- "**Model Plausibility.** If the diagnostics in situation 1 are better than in situation 3, but the diagnostics in situation 4 are better than in situation 2, then things are less clear and the justification of a change in estimators depends on the plausibility of the different assumptions about potential outcomes." (p.853)

The paper declines to rank them: "The normative value or relative ranking of these criteria should be left to individual research communities. Without a declared design, in particular the model and inquiry, none of these criteria can be evaluated, complicating the defense of claims for both the critic and the original author." (p.853). The criteria are stated in the context of replication and critique of a published study; "better" is not defined numerically and no diagnosand is singled out as the yardstick.

*Declaration timing, pre-registration, and reconciliation (pp.838-839, 851-852, 856; full verbatim passages are in the audit section below):*
- "Declare" is not "pre-register": "We emphasize that the term 'declare' does not imply a public declaration or even necessarily a declaration before research takes place. A researcher may declare the features of designs in our framework for their own understanding and declaring designs may be useful before or after the research is implemented." (p.839)
- Ex ante use: "Ex ante declarations can be used to improve designs and facilitate preregistration, analysis, and reconciliation of intended and actual analyses." (abstract, p.838). "Later, a researcher may include design declaration and diagnosis as part of a preanalysis plan or in a funding request." (p.839)
- What a pre-registration must fix: "The actions to be taken by researchers are described by the data strategy and the answer strategy; these two features of a design are clearly relevant elements of a preregistration document. In order to know which design choices were made ex ante and which were arrived at ex post, researchers need to communicate their data and answer strategies unambiguously. However, assessing whether the data and answer strategies are any good usually requires specifying a model and an inquiry." (p.852)
- The proposed adequacy test: "Rather than asking: 'are the boxes checked?' the question becomes: 'can it be diagnosed?'" (p.852)
- Reconciliation: "Declaration of a design in code also enables a final and infrequently practiced step of the registration process, in which the researcher 'reports and reconciles' the final with the planned analysis. Identifying how and whether the features of a design diverge between ex ante and ex post declarations highlights deviations from the preanalysis plan. The magnitude of such deviations determines whether results should be considered exploratory or confirmatory." (p.852)
- Reconciliation by code comparison: "Reconciliation of designs declared in code can be conducted automatically, by comparing changes to the code itself (e.g., a move from the use of a stratified sampling function to simple random sampling) and by comparing key variables in the design such as sample sizes." (p.852)
- No enforcement: "While MIDA may encourage more honest declarations, there is nothing in the framework that enforces them. As ever, garbage-in, garbage-out." (p.856)

*Observational designs (pp.848-849, 858):*
- Assignment process: "In designs in which the sampling process or the assignment process are in the control of researchers, p_Z is known. In observational designs, researchers either know or assume p_Z based on substantive knowledge." (p.858)
- The appendix frames its code walk-through as "an application in which the assignment procedure is known, as in an experimental or quasi-experimental design" (p.858).
- Inquiries should be stated as summaries of potential outcomes, not as parameters of the model used for estimation: "We run into a problem if we want to assess the properties of strategies under different assumptions about data generation when the inquiry itself depends on the data generating model. To address this problem, we can declare an inquiry as a summary of differences in potential outcomes across conditions" (pp.848-849). The conclusion says the framework "nudges them to make sure that their questions are defined clearly and *independently of their answer strategies*" (p.856).
- Printed estimand definitions in the observational regression example (p.849; unnumbered):

  ```
  tau == E_x E_i( Y_i(x) - Y_i(x - 1) )

  "we might define alpha and beta as the solutions to:"
  min over (alpha, beta) of   sum_i  integral ( Y_i(x) - alpha - beta * x )^2 f(x) dx

  Footnote 15 alternative:   tau == E[ Y_i(x_i) - Y_i(x_i - delta) ] / delta      for small delta
  ```

  "Here Y_i(x) is the (unknown) potential outcome for unit i in condition x. Estimand beta can be thought of as the coefficient one would get on x if one were to able to regress all possible potential outcomes on all possible conditions for all units (given density of interest f(x))." (p.849)
- Three observational worked examples are given (regression, matching on observables, regression discontinuity); see the audit section. Difference-in-differences, panel data and event studies do not appear.

*Limits and edge cases stated by the paper:*
- **Diagnoses are only as good as the declared conjectures.** "Given that design declaration includes declarations of conjectures about the world it is possible to choose inputs so that a design passes any diagnostic test set for it." (p.856)
- **Simulation results are local.** "a simulation-based claim to unbiasedness that incorporates all features of a design is still only good with respect to the precise conditions of the simulation (in contrast, analytic results, when available, may extend over general classes of designs)" (p.856)
- **The null-effect model cannot support every diagnosand.** A null potential outcomes function "will not work for diagnosands like power or the Type-S rate" (p.858).
- **Estimands that are not averages over sample units.** In regression discontinuity "the estimand here is not an average of potential outcomes of a set of sample units, but rather an unobservable quantity defined at the limit of the discontinuity. This feature makes the definition of diagnosands such as bias or external validity conceptually difficult." (p.849)
- **Undefined estimands.** In the replication example, "the weight of infants in control areas whose lives would have been saved if they had been in the treatment is undefined" (p.853); the inquiry had to be redefined for a subgroup.
- **Designs that cannot be declared.** "For more inductive research, by contrast, researchers may simply not have enough information about possible quantities of interest to declare a design in advance." (p.839). "When a design cannot be declared, non-declarability is all the framework provides, and in such cases we urge caution in drawing conclusions about design quality." (p.856)
- **Not everything is quantifiable.** "although many aspects of design quality can be assessed through design diagnosis, many cannot. For instance the contribution to an academic literature, relevance to a policy decision, and impact on public debate are unlikely to be quantifiable ex ante." (p.839)
- **Estimator failures, non-convergence, missing data, simulation seeds:** Not discussed in paper.

**Reference implementation(s):** DeclareDesign, "an accompanying open-source software package" (p.838), cited as "Blair, Graeme, Jasper Cooper, Alexander Coppock, Macartan Humphreys, and Neal Fultz. 2018. 'DeclareDesign.' Software package for R, available at http://declaredesign.org." (p.856). Function names printed in the paper are listed under "Software as described in the paper" in the audit section. "Replication materials can be found on Dataverse at: https://doi.org/10.7910/DVN/XYT1VB." (p.856). Neither the package nor the replication archive was examined in this review.

### Paper-derived requirements

*Only items the paper specifies or explicitly recommends. These constrain what may be attributed to this source; they do not constrain the API surface.*

- [ ] A design is declared as four elements `<M, I, D, A>`: model, inquiry, data strategy, answer strategy (pp.839-840)
- [ ] The model contains a signature (variables and their ranges), functional relations / potential outcomes, and a distribution over exogenous variables `P(U)` (pp.839-840)
- [ ] The inquiry defines an estimand `a^M` whose value is computed under the model, separately from the answer strategy (pp.840, 856)
- [ ] The data strategy covers sampling, assignment and measurement (p.840)
- [ ] The answer strategy may use realized data and information from the design, but has no access to the full schedule of potential outcomes ("functions that use information from realized data and the design", p.858), and returns the statistics needed for the diagnostic statistics (estimate, standard error, p-value, confidence interval in the paper's example) (pp.858-859)
- [ ] A diagnostic statistic is computed per run; a diagnosand is a summary of its distribution across runs (p.840)
- [ ] A declaration is "diagnosand-complete" for a diagnosand only if that diagnosand can be calculated from it; completeness is assessed per diagnosand (p.842; Table 1, p.841)
- [ ] Power needs a p-value and a significance threshold but no estimand; bias, RMSE and coverage need an estimand (p.842; Table 1)
- [ ] Diagnosis is by Monte Carlo: run the design `m` times, each run redrawing the population, potential outcomes, estimand, data and answer, then summarize (Table 3, p.845)
- [ ] A declaration describes a data generating process and a sampling strategy, not a particular realization (p.858)
- [ ] Recommended: quantify simulation uncertainty in each diagnosand estimate with a nonparametric bootstrap and report the diagnosand with its standard error (p.845; footnote 11)
- [ ] Recommended: assess the sensitivity of the diagnosis to alternative models and strategies, including alternative `P(U)` (pp.846, 858)
- [ ] Recommended: consider a null potential outcomes function and a series of potential outcomes functions matching competing theories (p.858)
- [ ] Designs are compared diagnosand by diagnosand against alternatives that vary the data strategy, the answer strategy, or the model (pp.840-841, 851-852)
- [ ] More than power is to be examined; bias, coverage and RMSE are named as also important (pp.841, 844, 856)
- [ ] For pre-registration, the data strategy and answer strategy are communicated unambiguously, with a model and inquiry so that the design can be diagnosed (p.852)
- [ ] Ex ante and ex post declarations are compared to "report and reconcile" the planned and final analysis; deviations are highlighted (p.852)
- [ ] Optional: the three named criteria for admitting an alternative answer strategy - home ground dominance, robustness to alternative models, model plausibility (p.853)

### What the paper does NOT provide

**The paper supplies a framework, a vocabulary, and a simulation procedure over a fully assumed model.** Each item below was checked against the full text of pp.838-859.

- **(a) No placebo assignment or resampling on real untreated data.** Absent. Every diagnosis in the paper simulates from an assumed model. The closest statements are that researchers "can inform their substantive models with existing data, such as baseline surveys" (p.846), that they "may even have an existing data set of the units that will be in the study with background characteristics" (p.858), and the suggestion of a "null potential outcomes function" (p.858), which is a simulated zero effect in an assumed model. The word "placebo" does not appear. The nonparametric bootstrap in the paper is applied to simulation output to obtain the standard error of a diagnosand, not to real data.
- **(b) No injection of effects into real data.** Absent. Effects enter only through the declared potential outcomes function of the assumed model (for example `Y ~ 0.25 * Z + u`, p.858).
- **(c) No difference-in-differences or staggered-adoption designs.** Absent. Neither term appears. No panel data, repeated-measures, pre/post, event-study or parallel-trends content appears. The observational examples are regression, matching on observables, and regression discontinuity (pp.848-849). "Quasi-experimental" appears three times in the text without elaboration (pp.842, 845, 858) and once in a reference title.
- **(d) No few-cluster inference.** Absent. Clusters appear only as a feature of the replication example (cluster assignment in "25 of the 50 catchment areas", p.854; "187 clusters instead of 50", p.854) and as an input to power calculators ("the number of units or clusters", p.843). No small-number-of-clusters inference method or diagnosis is discussed.
- **(e) No minimum detectable effect as a diagnosand.** Absent. The phrase occurs once, in an analogy about what power calculators focus attention on (p.846). It is not in Table 1, has no definition, and no search procedure is given.
- **(f) No held-out or split-sample validation of a design chosen by diagnosis.** Absent. The paper does diagnose a split-sample ANALYSIS procedure (explore on half the data, test on the other half; pp.850-851), but that is an answer strategy being diagnosed inside an assumed model. It is not a validation of an estimator or design that was selected by diagnosis, and no such validation step is described.
- **(g) No mechanical lock or enforcement mechanism.** Absent, and the paper says so: "there is nothing in the framework that enforces them" (p.856). What is offered is declaration, communication and reporting: a design declared in code, a pre-registration that states the data and answer strategies, and a comparison of ex ante and ex post declarations that "can be conducted automatically, by comparing changes to the code itself" (p.852). That comparison detects and reports deviations; it does not prevent them. The paper also states that "declare" does not imply a public or prior declaration (p.839).
- **(h) No rule for ranking or selecting among estimators.** Absent as a rule. The paper names three criteria for admitting an alternative answer strategy (p.853) but states that their "normative value or relative ranking ... should be left to individual research communities" (p.853), defines no numerical "better", and calls overall evaluation "multi-dimensional and qualitative" (p.841). No composite score, threshold, tie-break, or selection algorithm is given.

Also absent:

- **No size / false-positive-rate diagnosand.** Rejection under a true null is not named as a diagnosand; "Type I errors" are mentioned once in passing (p.850).
- **No mean-standard-error or standard-error-calibration diagnosand.** The standard error of the estimate is never summarized as a diagnosand.
- **No formula for most diagnosands.** Only bias, power, mean squared error and the expected answer have a printed expression; the rest of Table 1 is verbal.
- **No recommended number of simulations**, no convergence criterion beyond comparing the bootstrap uncertainty with a reference value, and no number of bootstrap replicates.
- **No fixed-population (finite-sample-conditional) diagnosis.** The printed procedure redraws the population in every run.
- **No guidance on choosing an effect size** beyond naming the need, sensitivity analysis, and the two approaches on p.858.
- **No treatment of estimator failures or missing draws** in a diagnosis.
- **No sample-size search procedure.** One example compares two cluster counts.
- **No general rule mapping study purposes to diagnosands.** The paper offers examples only: for policy-impact research it suggests an out-of-sample diagnosand, "such as bias relative to the population average treatment effect", and "the probability of making the right policy decision after research is conducted" (pp.838-839). Its conclusion states the limit: "As MIDA is silent on the question of a study's purpose, it cannot guide researchers or critics to the appropriate set of diagnosands by which to evaluate a design." (p.856)

**Any of these in a library implementation would be a library design choice requiring its own source. It would NOT be derived from this paper and must not be attributed to it.**

---

<!-- ============================================================
     END of registry-candidate block. Everything BELOW this
     marker is audit material (worked examples, verbatim quotes,
     implementation notes, gaps) and is NOT part of the proposed
     REGISTRY.md text.
     ============================================================ -->

## Worked Examples and Results

Every worked example in the paper is listed. For the main-text examples the formal declarations are in the Supplementary Materials, which were not reviewed, so the M / I / D / A descriptions below are the paper's main-text prose. The exception is the two-arm example, which is declared in code directly in Table 3 (p.845) and in the appendix (pp.858-859), both reviewed here.

| # | Example | Location | Design as described | What it illustrates |
|---|---|---|---|---|
| 1 | Confounded regression | pp.840-841; Supp. Section 1 | M: variables X, Y, Z with "Y = bX + Z + eps_Y; X = Z + eps_X". I: "what would be the average effect of a unit increase in X on Y in the population?". D: "data are gathered on X and Y for n randomly selected units". A: ordinary least squares | The bias diagnosand; a design can be poor "because the data and analysis strategy do not handle the confounding described by the model"; alternatives `D'` (randomly assign X) and `A'` (condition on Z) |
| 2 | Census of design tools on a blocked experiment | p.843; Table 2; Supp. Section 2 | "a common experimental design, in which assignment probabilities are heterogeneous by block"; three answer strategies: difference-in-means (DIM), block fixed effects (BFE), block fixed effects with inverse-probability weighting (written both "BFE-IPW" and "IPW-BFE") | Existing tools cannot declare most design elements; power alone misleads |
| 3 | Survey design with likely voters | p.846; Supp. Section 3.1 | M: latent voters "overstate their true propensity to vote". I: "true underlying support for the candidate among those who will vote". D: random sample plus vote-intention and turnout questions. A: support among likely voters | Bias from misreporting; "we can calibrate how far off our estimates will be under reasonable models of misreporting" |
| 4 | Bayesian descriptive inference | p.846; Supp. Section 3.2 | M: latent success probability per unit, one binomial draw each. I: the true latent probability. D: "a random sample of relatively few units". A: uniform priors (mean 0.50, SD 0.29) vs priors with more mass at 0.50 (SD 0.11) | Bayesian diagnosands; "the informative priors decrease the posterior standard deviation by 40% relative to the uniform priors, but increase the bias by 33%" |
| 5 | Process tracing | pp.846-847; Supp. Section 3.3 | I: "whether X in fact caused Y in the specific case under analysis". D: select one case with X and Y present, collect two causal process observations with Pr(E1\|H) = 0.75, Pr(E1\|not H) = 0.25, Pr(E2\|H) = 0.30, Pr(E2\|not H) = 0.05 (combining the two also requires declaring their dependence: "the learning depends on the ways in which these CPOs are correlated", p.847). A: Bayes' rule | A researcher relying on the weaker "straw-in-the-wind" test "would make *better* inferences on average than one who relied solely on the 'smoking gun' test"; "tradeoffs involved in committing to one or another CPO strategy ex ante" |
| 6 | Qualitative comparative analysis | pp.847-848; Supp. Section 3.4 | M: Y arises iff cause A absent and cause B present. I: "the true minimal set of configurations of conditions that are sufficient to cause Y". D: truth table with measurement error. A: QMC vs CCubes minimization (both using the "parsimonious" solution, footnote 14) vs saturated regression (indicators for "all feasible configurations of A and B", selecting configurations that predict Y with probability "greater than 0.5") | Choice of algorithm matters little in simple cases; regression can target a QCA estimand |
| 7 | Nested mixed methods | p.848; Supp. Section 3.5 | Binary variables; inquiry on the X-Y relationship; integrated large-N and within-case analysis following Lieberman (2005) | Highlights "the design features that are left to researchers"; interactions between how many cases to sample at each stage and success thresholds |
| 8 | Observational regression | pp.848-849; Supp. Section 3.6 | M: y a nonlinear function of x. I: a linear summary of potential outcomes (`beta` as defined above). D: "passive observation of units in the population". A: OLS | "under uniform random assignment of x, the linear regression returns an unbiased estimate of a (linear) estimand, even though the true data generating process is nonlinear"; "unbiasedness is lost in a design in which different values of x_i are assigned with differing probabilities"; "without defining I, it is hard to see the conditions under which A is biased or unbiased" |
| 9 | Matching on observables | p.849; Supp. Section 3.7 | M: "three observable random variables are combined in a probit process that assigns the treatment variable, Z". I: "the average treatment effect of Z on the outcome Y among those actually assigned to treatment". A: matching "that tries to reconstruct the assignment process" vs naive difference-in-means | "matching improves mean-squared-error (E[(a^A - a^M)^2]) relative to a naive difference-in-means estimator of the treatment effect on the treated (ATT), but can nevertheless remain biased (E[a^A - a^M] != 0) if the matching algorithm does not successfully pair units with equal probabilities of assignment"; separates beliefs about M from details of A |
| 10 | Regression discontinuity | p.849; Supp. Section 3.8 | M: "unknown potential outcomes functions mapping average outcomes to the running and treatment variables" (the paper notes that "causal identification is often premised on the claim that potential outcomes are continuous at a critical threshold", p.849). I: "the difference in the conditional expectations of the two potential outcomes functions at the discontinuity". D: passive observation. A: "polynomial regression in which the assignment variable is linearly interacted with a fourth order polynomial transformation of the running variable" | The estimand is "an unobservable quantity defined at the limit of the discontinuity", which makes diagnosands such as bias "conceptually difficult" |
| 11 | Factorial vs three-arm experiment | pp.849-850; Figure 1; Supp. Section 3.9 | Models M_1..M_K with interaction from -0.2 to +0.2. I: "the average treatment effect of treatment 1 given all units are in the control condition for treatment 2". D: three arms with probability 1/3 each vs four factor combinations with probability 1/4 each. A: "a regression of the outcome on both treatment indicators with no interaction term included" | Comparing two data strategies across a grid of models; bias-variance tradeoff; diagnosands bias, RMSE, power |
| 12 | Discovery with a split sample | pp.850-851; Supp. Section 3.10 | M: a treatment with possibly heterogeneous effects by group. D: random assignment. A: examine heterogeneity "inside a random subgroup of the data. If they find evidence of differential effects they specify a new inquiry which is assessed on the remaining data", vs "a strategy that simply reports discoveries found using complete data ... (we call this the 'unprincipled' approach)" | "lower bias from principled discovery than from unprincipled discovery"; "Mean squared error is not necessarily lower for the principled approach since fewer data are used in the final test"; the principled strategy "is somewhat less likely to produce a result *at all*" |
| 13 | Modeling a data generating process | p.851 (no supplement section cited) | The estimand is a function; either "identify the ultimate quantities of interest *ex ante* and ... treat these as the estimands", or posit "a true underlying DGP as part of m" and estimate an approximating function | Diagnosands for model-focused work: compare fitted and true functions over a range of X, or use goodness of fit "such as the R^2" |
| 14 | Design replication of Bjorkman and Svensson (2009) | pp.853-855; Figure 2; Supp. Section 4 | See below | "Design replication"; ill-defined inquiry; exaggeration ratio; comparing four answer strategies under two models with the home-ground-dominance and robustness criteria |
| 15 | Two-arm design in code | pp.858-859 (appendix); Table 3, p.845 | N = 1000 population (N = 200 in Table 3), `Y ~ 0.25 * Z + u` (`Y ~ Z + u` in Table 3), sample n = 100, assign m = 50, difference-in-means, ATE estimand | How each element is declared in code; the bias diagnosand |

### Table 2 (p.843): "Existing Tools Cannot Declare Many Core Elements of Designs and, as a Result, Can Only Calculate Some Diagnosands"

| (a) Declare design elements | | | (b) Diagnosis capabilities | |
|---|---|---|---|---|
| (M) | Effect and block size correlated | 0/30 | Power (DIM estimator) | 28/30 |
| (I) | Estimand | 0/30 | Power (BFE estimator) | 13/30 |
| (D) | Sampling procedure | 0/30 | Power (IPW-BFE estimator) | 0/30 |
| (D) | Assignment procedure | 0/30 | Bias (*any* estimator) | 0/30 |
| (D) | Block sizes vary | 1/30 | Coverage (*any* estimator) | 0/30 |
| (A) | Probability weighting | 0/30 | SD of estimates (*any* estimator) | 0/30 |

Table note, verbatim: "Panel (a) indicates the number of tools that allow declaration of a particular feature of the design as part of the diagnosis. In the first row, for example, 0/30 indicates that no tool allows researchers to declare correlated effect and block sizes. Panel (b) indicates the number of tools that can perform a particular diagnosis. Results correspond to design tool census concluded in July 2017 and do not include tools published since then."

- Census scope (footnote 5, p.843): tools from four literature reviews plus "the first thirty results from Google searches" of three terms; "Thirty of the 143 tools we identified were able to diagnose inferential properties of designs, such as their power."
- Finding: "While the IPW-BFE estimator is better powered and less biased than the BFE estimator, its purported efficiency is misleading. IPW-BFE is better powered than DIM and BFE because it produces biased variance estimates that lead to a coverage probability that is too low. In terms of RMSE and the standard deviation of estimates, the IPW-BFE strategy does not outperform the BFE estimator." (p.843). No numbers are printed for these diagnosands.

### Figure 1 (p.850): "Diagnoses of Designs With Factorial or Three-Arm Assignment Strategies Illustrate a Bias-Variance Tradeoff"

Three panels (Bias, RMSE, Power) against "Interaction Effect Size" from -0.2 to 0.2. Caption, verbatim: "Bias (left), root mean-squared-error (center), and power (right) are displayed for two assignment strategies, a 2 x 2 treatment arm factorial design (black solid lines; circles) and a three-arm design (gray dashed lines; triangles) according to varying interaction effect sizes specified in the potential outcomes function (x axis). The third panel also shows power for the interaction effect (squares) from the factorial design."

- Text findings: "neither design exhibits bias when the true interaction term is equal to zero"; "when the interaction between the two treatments is stronger, the factorial design renders estimates of the effect of treatment 1 that are more and more biased relative to the 'pure' main effect estimand"; "When the interaction term is close to zero, the factorial design is preferred, because it is more powerful: it compares one half of the subject pool to the other half, whereas the three-arm design only compares a third to a third"; "When the true interaction between treatments is large, the three-arm design is then preferred." (p.850)
- Design guidance drawn: "if the scientific question of interest is the pure effect of each treatment, researchers should (perhaps counterintuitively) use a factorial design if they expect *weak* interaction effects." (p.850)
- **Visual approximation (the paper prints no numbers).** Bias: one series flat at 0; the other rises roughly linearly from about -0.10 at -0.2 to about +0.10 at +0.2. RMSE: one series flat at about 0.11; the second is U-shaped, about 0.14 at the ends and about 0.09 at 0; the squares series is flat at about 0.18. Power: one series flat at about 0.42; the second rises from about 0.2 to about 0.9; the squares series is about 0.2 at the ends and near 0.05 at 0.
- **Reviewer note - caption and legend disagree.** The caption assigns solid lines with circles to the factorial design and dashed lines with triangles to the three-arm design. The printed legend assigns the solid circles to "Three-arm Design" and the dashed triangles to "2x2 Factorial Design (Main Effect)". The text (the factorial design is the one whose bias grows with the interaction) is consistent with the legend, not the caption.

### Design replication of Bjorkman and Svensson (2009) (pp.853-855; Figure 2, p.854)

- Purpose: "using available information, we posit a Model, Inquiry, Data, and Answer strategy to assess properties of Bjorkman and Svensson (2009)" (p.853). "The exercise serves three purposes: first, it sheds light on the sorts of insights the design can produce without using the original study's data or code; second, it highlights how difficulties can arise from designs in which the inquiry is not well-defined; third, we can assess the properties of replication strategies" (p.853).
- Model: "unobserved variables, 'family health' and 'community health,' determine both whether infants survive early childhood and whether they are malnourished" (p.853).
- Inquiry, as printed (pp.853-854): `E[Weight(Z = 1) - Weight(Z = 0) | Alive(Z = 0) = Alive(Z = 1) = 1]`. The rejected alternative (footnote 18, p.854): `E[Weight(Z = 1) | Alive(Z = 1) = 1] - E[Weight(Z = 0) | Alive(Z = 0) = 1]`, which "would lead to very aberrant conclusions".
- Data strategy: "we stratify sampling on catchment area and cluster-assign households in 25 of the 50 catchment areas to the intervention" (p.854).
- Figure 2 ("Data-independent Replication of Estimates in Bjorkman and Svensson (2009)"): caption "Histograms display the frequency of simulated estimates of the effect of community monitoring on infant mortality (left) and on weight-for-age (right). The dashed vertical line shows the average estimate, the dotted vertical line shows the average estimand." Finding: "Under our postulated model of the world, the estimates of the effect on weight-for-age are biased downwards because it is precisely those infants with low health outcomes whose lives are saved by the treatment." (p.854). No numeric bias is printed and the number of simulations is not stated.
- Alternative answer strategy ("attempt to subset" to infants whose survival does not depend on treatment; the paper's candidate subgroup is children above age two, with the qualification "If indeed the survival of infants above this age threshold is unaffected by the treatment, then it is possible to provide unbiased estimates of the weight-for-age effect, if only among this group", p.854): "In terms of bias, such an approach does at least as well if we assume that there is no correlation between weight and mortality, and better if such a correlation does exist. It thus satisfies the 'robustness to alternative models' criterion." (p.854). Counter: "the alternative answer strategy does not meet the criterion of 'home ground dominance' with respect to RMSE. The increase in variance from subsetting to a smaller group may outweigh the bias reduction that it entails." (p.854)
- Exaggeration ratio: "on average, statistically significant estimates tend to exaggerate the true effect of the intervention on mortality by a factor of two and on weight-for-age by a factor of four. In other words, even though the study estimates effects on mortality in an unbiased manner, limiting attention to statistically significant effects provides estimates that are twice as large in absolute value as the true effect size on average. By contrast, using the same sample size as that employed in Raffler, Posner, and Parkerson (2019) reduces the exaggeration ratio on the mortality estimand to where it should be, around one." (p.855)
- Four answer strategies under two models (p.855): ignore prior presence of community-based organizations (CBO); include it irrespective of imbalance; include it "only if the CBO presence is significantly imbalanced among the 50 treatment and control clusters at the alpha = 0.05 level"; include it and its interaction with treatment. Models: CBO presence unrelated to health outcomes vs highly correlated.
  - "Including the interaction term is a strictly dominated strategy from the standpoint of reducing mean squared error: irrespective of whether CBO presence is correlated with health outcomes or imbalanced, the RMSE expected under this strategy is higher than under any other strategy." (p.855)
  - "when there is a correlation between health outcomes and CBO presence, strategies that include CBO presence improve RMSE considerably, especially when there is imbalance." (p.855)
  - Conclusion drawn: "a diagnostic approach to replication clarifies that one should resist updating beliefs about the study based on the use of interaction terms, but that the inclusion of the CBO indicator only harms inferences in a very small subset of cases." (p.855)
  - No RMSE values are printed.

### Software as described in the paper

All statements are from Table 3 (p.845) and the appendix (pp.858-859). Nothing here is taken from outside the paper.

| Function | Paper's label | What the paper says it does | Printed call(s) |
|---|---|---|---|
| `declare_population` | "Declare background variables" (M); "The population" | "Defines the population variables, including both observed and unobserved X." | `declare_population(N = 200, u = rnorm(N))`; `declare_population(N = 1000, u = rnorm(N))` |
| `declare_potential_outcomes` | "Declare functional relations" (M); "The structural equations, or potential outcomes function" | "defines conjectured potential outcomes given interventions Z and parents"; "maps from a treatment condition vector (Z) and background data u, generated by P(U) to a vector of outcomes" | `declare_potential_outcomes(Y ~ Z + u)`; `declare_potential_outcomes(Y ~ 0.25 * Z + u)` |
| `declare_estimand` | "Declare inquiry" (I); "Estimands" | "The estimand function creates a summary of potential outcomes. In principle, the estimand function can also take realizations of assignments as arguments, in order to calculate post-treatment estimands." | `declare_estimand(ATE = mean(Y_Z_1 - Y_Z_0))` |
| `declare_sampling` | "Declare sampling" (D); "The sampling strategy" | "Defines the distribution over possible samples for which outcomes are measured, p_S." | `declare_sampling(n = 100)` |
| `declare_assignment` | "Declare assignment" (D); "The treatment assignment strategy" | "Defines the strategy for assigning variables under the notional control of researchers." | `declare_assignment(m = 50)` |
| `declare_reveal` | "Declare outcome revelation" (D) | "an additional step in which the outcome for Y is revealed after Z is determined"; "reveals the outcome Y on the basis of the potential outcomes and a realized random assignment" | `declare_reveal(Y, Z)` |
| `declare_estimator` | "Declare answer strategy" (A) | "an estimator function takes data and returns an estimate of a treatment effect using the difference-in-means estimator, as well as a set of associated statistics, including the standard error, p-value, and the confidence interval"; "we associate estimators with estimands" | `declare_estimator(Y ~ Z, estimand = "ATE")`; `declare_estimator(Y ~ Z, model = lm_robust, estimand = "ATE")` |
| `+` operator | "Declare design, `<M, I, D, A>`" | "The design is the concatenation of these, which we represent using the '+' operator"; "Order matters." | `design <- p_U + f_Y + I + p_S + p_Z + R + A` |
| `declare_diagnosands` | "Declare a diagnosand" | Defines the diagnostic statistic and diagnosand; "the two can be coded as a single function" | `declare_diagnosands(bias = mean(estimate - estimand))` |
| `diagnose_design` | "Calculate a diagnosand" | "Diagnosing the design involves simulating the design many times, then calculating the value of the diagnosand from the resulting simulations"; "returns an estimate of the diagnosands, along with other metadata associated with the simulations" | `diagnose_design(design, diagnosands = bias, sims = m)`; `diagnose_design(design = design, diagnosands = diagnosand, sims = 500, bootstrap_sims = FALSE)` |

- "Formally, each of these steps is a function." (p.845)
- "These six features represent the study." (p.859), referring to the appendix labels M(1), M(2), I, D(1), D(2), A.
- The language of the package ("Software package for R") is given only in the reference-list entry (p.856).
- Argument meanings (`N`, `n`, `m`, `sims`, `bootstrap_sims`, `model`, `estimand`) are not defined beyond the accompanying prose. The prose for the appendix says "each unit generated by P(U) is sampled with 10% probability" and "each sampled unit is assigned to treatment independently with probability 0.5" (p.858). Reviewer arithmetic: 100 / 1000 = 10% and 50 / 100 = 0.5; whether the printed arguments fix counts or probabilities is not stated.

---

## Pre-registration, Transparency, and Researcher Discretion (verbatim)

Everything the paper says on declaring before outcomes are seen, committing to an answer strategy, reconciling planned and realized designs, and selective reporting. All quotes verbatim with journal page numbers.

1. **Declaration is not, by itself, pre-registration.**
   - "We emphasize that the term 'declare' does not imply a public declaration or even necessarily a declaration before research takes place. A researcher may declare the features of designs in our framework for their own understanding and declaring designs may be useful before or after the research is implemented." (p.839)
   - "Even if declared ex-post, formal declaration still has benefits. The complete characterization can help readers understand the properties of a research project, facilitate transparent replication, and can help guide future (re-)analysis of the study data." (p.839)
   - "Ex ante declarations can be used to improve designs and facilitate preregistration, analysis, and reconciliation of intended and actual analyses. Ex post declarations are useful for describing, sharing, reanalyzing, and critiquing existing designs." (abstract, p.838)

2. **Ex ante declaration and preanalysis plans.**
   - "The formal characterization and diagnosis of designs before implementation can serve many purposes. First, researchers can learn about and improve their inferential strategies. Done at this stage, diagnosis of a design and alternatives can help a researcher select from a range of designs, conditional upon beliefs about the world. Later, a researcher may include design declaration and diagnosis as part of a preanalysis plan or in a funding request. At this stage, the full specification of a design serves a communication function and enables third parties to understand a design and an author's intentions." (p.839)
   - "In prospective confirmatory work, for example, researchers may have access to all design-relevant information prior to launching their study. For more inductive research, by contrast, researchers may simply not have enough information about possible quantities of interest to declare a design in advance." (p.839)

3. **Researcher discretion and frontloading design decisions.**
   - "The move toward increasing credibility of research in the social sciences places a premium on considering alternative data strategies and analysis strategies at early stages of research projects, not only because it reduces researcher discretion after observing outcomes, but more importantly because it can improve the quality of the final research design. While there is nothing new about the idea of determining features such as sampling and estimation strategies ex ante, in practice many designs are finalized late in the research process, after data are collected. Frontloading design decisions is difficult not only because existing tools are rudimentary and often misleading, but because it is not clear in current practice what features of a design must be considered ex ante." (p.851)
   - "We provide a framework for identifying *which* features affect the assessment of a design's properties, declaring designs and diagnosing their inferential quality, and frontloading design decisions. ... Researchers can undertake each step before study implementation or data collection." (pp.851-852)

4. **Selective reporting and pre-registration.**
   - "Bias in published results can arise for many reasons. For example, researchers may deliberately or inadvertently select analysis strategies because they produce statistically significant results. Proposed solutions to reduce this kind of bias focus on various types of preregistration of analysis strategies by researchers (Casey, Glennerster, and Miguel 2012; Green and Lin 2016; Nosek et al. 2015; Rennie 2004; Zarin and Tse 2008). Study registries are now operating in numerous areas of social science, including those hosted by the American Economic Association, Evidence in Governance and Politics, and the Center for Open Science. Bias may also arise from reviewers basing publication recommendations on statistical significance. Results-blind review processes are being introduced in some journals to address this form of bias (e.g., Findley et al. 2016)." (p.852)
   - "However, the effectiveness of design registries and results-blind review in reducing the scope for either form of publication bias depends on clarity over which elements must be included to describe the design. In practice, some registries rely on checklists and preanalysis plans exhibit great variation, ranging from lists of written hypotheses to all-but-results journal articles. In our view, the solution to this problem does not lie in ever-more-specific questionnaires, but rather in a new way of characterizing designs whose analytic features can be diagnosed through simulation." (p.852)

5. **What must be specified to pre-register effectively.**
   - "The actions to be taken by researchers are described by the data strategy and the answer strategy; these two features of a design are clearly relevant elements of a preregistration document. In order to know which design choices were made ex ante and which were arrived at ex post, researchers need to communicate their data and answer strategies unambiguously. However, assessing whether the data and answer strategies are any good usually requires specifying a model and an inquiry. Design declaration can clarify for researchers and third parties what aspects of a study need to be specified in order to meet standards for effective preregistration. Rather than asking: 'are the boxes checked?' the question becomes: 'can it be diagnosed?' The relevant diagnosands will likely depend on the type of research design. However, if an experimental design is, for example, 'bias complete,' then we know that sufficient information has been given to define the question, data, and answer strategy unambiguously." (p.852)

6. **Reconciling the planned and the realized design; reporting deviations.**
   - "Declaration of a design in code also enables a final and infrequently practiced step of the registration process, in which the researcher 'reports and reconciles' the final with the planned analysis. Identifying how and whether the features of a design diverge between ex ante and ex post declarations highlights deviations from the preanalysis plan. The magnitude of such deviations determines whether results should be considered exploratory or confirmatory. At present, this exercise requires a review of dozens of pages of text, such that differences (or similarities) are not immediately clear even to close readers. Reconciliation of designs declared in code can be conducted automatically, by comparing changes to the code itself (e.g., a move from the use of a stratified sampling function to simple random sampling) and by comparing key variables in the design such as sample sizes." (p.852)

7. **Inductive work and guarding against false discovery.**
   - "Split-sample procedures such as this enable researchers to learn about the data inductively while protecting against Type I errors (for an early discussion of the design, see Cox 1975)." (p.850)
   - "Formalizing this kind of process-tracing exercise leads to non-obvious insights about the tradeoffs involved in committing to one or another CPO strategy ex ante." (p.847)
   - "One asks the researcher to identify the ultimate quantities of interest *ex ante* and to treat these as the estimands." (p.851)

8. **Replication, critique, and intentions.**
   - "Replication-whether verification, reanalysis of the original data, or reproduction using fresh studies-provides incentives for researchers to be clear and transparent in their analysis strategies, and can build confidence in findings." (p.852)
   - "A standard practice for replicators engaging in reanalysis is to propose a range of alternative strategies and assess the robustness of the *data-dependent* estimates to different analyses. The problem with this approach is that, when divergent results are found, third parties do not have clear grounds to decide which results to believe. This issue is compounded by the fact that, in changing the analysis strategy, replicators risk departing from the estimand of the original study, possibly providing different answers to different questions." (p.852)
   - "In a design replication, a scholar restates the essential design characteristics to learn about what the study *could have* revealed, not just what the original author reports *was* revealed. ... By emphasizing abstract properties of the design, design replication provides grounds to support alternative analyses on the basis of the original authors' intentions and not on the basis of the degree of divergence of results. Conversely, it provides authors with grounds to question claims made by their critics." (p.852)

9. **No enforcement; gaming is possible.**
   - "Given that design declaration includes declarations of conjectures about the world it is possible to choose inputs so that a design passes any diagnostic test set for it." (p.856)
   - "Still worse, simulation parameters might be selected because of their properties. A power analysis, for instance, may be useless if implausible parameters are chosen to raise power artificially. While MIDA may encourage more honest declarations, there is nothing in the framework that enforces them. As ever, garbage-in, garbage-out." (p.856)

The paper does not use the terms "researcher degrees of freedom", "fishing", "p-hacking", "forking paths" or "specification search". Its vocabulary for the problem is "researcher discretion after observing outcomes" (p.851), researchers who "deliberately or inadvertently select analysis strategies because they produce statistically significant results" (p.852), and "publication bias" (p.852). It gives no procedure for what a researcher should do after a deviation beyond identifying and reporting it, and no threshold for the "magnitude" that separates exploratory from confirmatory.

---

## Limitations Stated by the Authors

The conclusion lists "four concerns" (pp.855-856), introduced with: "Designing high quality research is difficult and comes with many pitfalls, only a subset of which are ameliorated by the MIDA framework. Others we fail to address entirely and in some cases, we may even exacerbate them." (pp.855-856)

1. **Meaningless diagnoses.** "The first is the worry that evaluative weight could get placed on essentially meaningless diagnoses." (p.856). The supporting sentences are quoted in item 9 of the previous section and under "Limits and edge cases".
2. **Narrow or inappropriate diagnosands.** "Second, we see a risk that research may get evaluated on the basis of a narrow, but perhaps inappropriate set of diagnosands. Statistical power is often invoked as a key design feature-but there may be little value in knowing the power of a study that is biased away from its target of inference. The appropriateness of the diagnosand depends on the purposes of the study. As MIDA is silent on the question of a study's purpose, it cannot guide researchers or critics to the appropriate set of diagnosands by which to evaluate a design. An advantage of the approach is that the choice of diagnosands gets highlighted and new diagnosands can be generated in response to substantive concerns." (p.856)
3. **Statistical properties can crowd out substance.** "Third, emphasis on the statistical properties of a design can obscure the substantive importance of a question being answered or other qualitative features of a design. ... Our framework can help researchers determine whether a particular design answers a question well (or at all), and it also nudges them to make sure that their questions are defined clearly and *independently of their answer strategies*. It cannot, however, help researchers choose good questions." (p.856)
4. **Declarability is not quality.** "Finally, we see a risk that the variation in the suitability of design declaration to different research strategies may be taken as evidence of the relative superiority of different types of research strategies. While we believe that the range of strategies that can be declared and diagnosed is wider than what one might at first think possible, there is no reason to believe that all strong designs can be declared either ex ante or ex post." (p.856)

Other caveats stated elsewhere:
- "We acknowledge from the outset that variation in research strategy limits the utility of our procedure for different types of research." (p.839)
- Ex post declaration can fail: "in others it may not be possible to fully reconstruct the inferential procedure after the fact. For instance, although researchers might be able to provide compelling grounds for their inferences, they may not be able to describe what inferences they would have drawn had different data been realized." (p.839)
- "Although not all approaches to process tracing are readily amenable to design declaration" (p.846).
- "Although our simulation approach has a frequentist flavor, the MIDA framework itself can also be applied to Bayesian strategies." (p.846)
- The bootstrap uncertainty procedure "depends on the researcher choosing a 'good' diagnosand estimator" (footnote 12, p.846).
- The tool census "concluded in July 2017 and do[es] not include tools published since then" (Table 2 note, p.843).
- Practical framing: "In the end, we are asking that scholars add a step to their workflow. We want scholars to formally declare and diagnose their research designs both in order to learn about them and to improve them." (p.856)

---

## Implementation Notes (audit notes - NOT registry-candidate)

### Data Structure Requirements

What the paper's procedure uses (not a proposed library contract):
- No data set is required. The inputs are declarations: a population generator `P(U)`, a potential outcomes function, an estimand function, sampling and assignment strategies, and an answer strategy (Table 3, p.845).
- An existing data set of the study units "with background characteristics" may inform `P(U)` (p.858), and existing data "such as baseline surveys" may inform the model (p.846). How such data would enter a declaration is not described.
- The answer strategy must return whatever the chosen diagnostic statistics need: an estimate for bias and RMSE, a p-value for power, a confidence interval for coverage (pp.842, 859).
- The estimand must be computable from the model (the full schedule of potential outcomes), which is unavailable to the answer strategy (pp.840, 858).
- Panel structure, period-indexed covariates, and treatment-adoption timing: Not developed in paper. Two nearby points the paper does make: the time between treatment and endline data collection may enter the model "if treatment effects decay" (p.840), and the replication example uses a cluster-level estimand and estimator for mortality ("We estimate mortality at the cluster level"; "The mortality estimand is defined at the cluster level", p.854).
- Missing data, attrition, unbalanced samples: Not discussed in paper (noncompliance is mentioned once as something that "can be included in many parts of the design", p.840).

### Computational Considerations

- One diagnosis is `m` independent runs of the six-step simulation (Table 3, p.845). Draws are "i.i.d. draws of the diagnostic statistic" by construction (footnote 12, p.846).
- "simulations are often computationally expensive" (p.845). No runtime, parallelization, or seeding guidance is given.
- Simulation count: `sims = m` (Table 3); `sims = 500` in the appendix example (p.859). No rule for choosing `m` other than checking the bootstrap uncertainty of the diagnosand against a reference value (pp.845-846).
- Bootstrap: nonparametric, over the simulation draws by implication (the paper says the sampling distribution "of the diagnosands" is estimated; it does not say what is resampled). Number of replicates not stated. Reviewer note: the only fully specified `diagnose_design` call in the paper passes `bootstrap_sims = FALSE`, so the recommended uncertainty step is not shown in use.
- Comparing designs across a grid of models multiplies the work: the factorial example evaluates two designs at several interaction values between -0.2 and +0.2 (Figure 1 shows nine x-axis points; reviewer visual count).
- Whether alternative designs are compared on common random numbers (paired draws) is not discussed in paper.
- Handling of runs in which the answer strategy fails: Not discussed in paper. For the two conditional diagnosands only significant runs enter by definition (Table 1, p.841); what to report when no run is significant: Not discussed in paper.

### Relation to Existing diff-diff Estimators

- **Existing simulation tools.** `diff_diff/power.py` has `simulate_power`, `simulate_mde`, and `simulate_sample_size`, which by default draw from an internal per-estimator registry of SYNTHETIC data generators and also accept a custom `data_generator` / `data_generator_kwargs` that bypasses the registry. Existing argument spellings: `estimator`, `n_simulations`, `alpha`, `effect_sizes`, `seed`, `data_generator`, `data_generator_kwargs`, `estimator_kwargs`, `result_extractor`, `progress`.
- **Existing result fields.** `SimulationPowerResults` already carries `power`, `power_se`, `power_ci`, `rejection_rate`, `mean_estimate`, `std_estimate`, `mean_se`, `coverage`, `bias`, `rmse`, `true_effect`, `effect_sizes`, `powers`, `alpha`, `n_simulations`, `n_simulation_failures`, `estimator_name`. `SimulationMDEResults` carries `mde`, `power_at_mde`, `target_power`.
- **Diagnosand-to-field map.** Matching is by concept and name only; whether each field's computation matches the paper's definition was not checked in this review.

  | Paper diagnosand | Paper definition location | Existing library field |
  |---|---|---|
  | Power | Table 1, p.841; `E(s)` with `s = 1(p <= 0.05)`, p.840; p.842 | `power` |
  | Estimation bias | Table 1, p.841; `E(e)`, p.840; `E[a^A - a^M]`, p.849 | `bias` |
  | Sampling bias | Table 1, p.841 | none |
  | RMSE | Table 1, p.841 | `rmse` |
  | Mean squared error | `E[(a^A - a^M)^2]`, p.849 | none (`rmse` is the root form) |
  | Coverage | Table 1, p.841 | `coverage` |
  | SD of estimates | Table 1, p.841 | `std_estimate` |
  | SD of estimands | Table 1, p.841 | none |
  | Imbalance | Table 1, p.841 | none |
  | Type S rate | Table 1, p.841 | none |
  | Exaggeration ratio | Table 1, p.841; p.855 | none |
  | Value for money | Table 1, p.841 | none |
  | Robustness | Table 1, p.841 | none |
  | Average answer / mean estimate | `E[a^A]`, p.854; Figure 2 | `mean_estimate` |
  | Mean estimand | Figure 2, p.854 | none (nearest existing name is `true_effect`; equivalence not established) |
  | Out-of-sample bias | p.838 | none |
  | Probability of the right policy decision | pp.838-839 | none |
  | Posterior shift / posterior standard deviation | p.846 | none |
  | Correct recovery of a configuration | p.848 | none |
  | Probability of producing a result | p.851 | none |
  | Model-fit diagnosands (function comparison, R^2) | p.851 | none |
  | Standard error of a diagnosand estimate (bootstrap) | p.845; footnote 11 | `power_se` and `power_ci` (by name, for power); none among the fields listed above for the other diagnosands |

- **Existing fields with no diagnosand of that name in the paper.** `mean_se` (the paper never summarizes the estimator's standard error as a diagnosand), `rejection_rate` (the paper names power only), `n_simulation_failures` (failed runs are not discussed in the paper), and `mde` / `power_at_mde` / `target_power` (minimum detectable effect is not a diagnosand in the paper; it is mentioned once on p.846).
- **Reviewer note on vocabulary overlap.** By name, the paper's `sims` corresponds to `n_simulations`, its significance threshold to `alpha`, its answer strategy to `estimator` / `estimator_kwargs`, and its model (population plus potential outcomes) to `data_generator` / `data_generator_kwargs` with `effect_sizes`. The paper's inquiry / estimand and its data strategy have no separately named counterpart in the argument list above.
- **Reviewer note on how the paper positions simulation.** The paper's diagnosis procedure is Monte Carlo simulation from a fully assumed model, which is the same kind of exercise as the existing synthetic-generator simulation functions. It supplies vocabulary and a diagnosand roster for that exercise. It supplies nothing for a diagnosis run on a user's own real data (no placebo assignment, no resampling, no effect injection), so it does not close the gap identified in the companion review of Coglianese and Fang (2026), whose source supplies a placebo MSE summary on real data and none of the diagnosands above.
- **Existing placebo diagnostics.** `diff_diff/diagnostics.py` has placebo diagnostics for the basic DiD estimator - individual fake-timing and fake-group placebo fits, a permutation procedure that repeatedly reassigns treatment, and a leave-one-out procedure that refits after dropping each treated unit (`run_placebo_test`, `placebo_timing_test`, `placebo_group_test`, `permutation_test`, `leave_one_out_test`, each returning `PlaceboTestResults`, plus `run_all_placebo_tests`, which returns a dictionary of those results or structured error dictionaries). The paper has no counterpart to these.
- **Naming constraint.** The library has a standing constraint to reuse existing method / argument / result-field names wherever a concept already exists, so the "none" rows above are open naming questions and the matched rows are not. The API itself is to be decided in the implementation plan, not here.

---

## Gaps and Uncertainties

**Diagnosand definitions**
- **Most diagnosands have no printed formula.** Only bias (`E(e)`, `E[a^A - a^M]`, `mean(estimate - estimand)`), power (`E(s)`), mean squared error (`E[(a^A - a^M)^2]`) and the expected answer (`E[a^A]`) are written out. RMSE, coverage, SD of estimates, SD of estimands, sampling bias, imbalance, Type S rate, exaggeration ratio, value for money and robustness are verbal one-liners in Table 1. The formulas in this review for those are reviewer reconstructions.
- **Power threshold.** p.840 uses `s = 1(p <= 0.05)`; p.842 says "lower than a critical value (e.g., 0.005)". Both are as printed. Whether 0.005 is intended or a misprint is not stated, and the inequality is weak on p.840 and strict in the wording on p.842.
- **Exaggeration ratio.** Table 1 says "ratio of absolute value of estimate to estimand" (p.841); p.855 says "the ratio of the absolute value of the estimate to the absolute value of the estimand, given that the estimated effect is significant". Together these specify the conditional expectation of the per-run ratio (a mean of ratios over significant runs); the only wording difference is whether the estimand in the denominator is taken in absolute value.
- **Type S rate.** Only statistically significant runs enter it, by definition (Table 1, p.841). The treatment of a zero estimand, and what to return when no simulated run is significant, are not specified. The appendix notes only that a null potential outcomes function "will not work" for it (p.858).
- **Coverage.** No standalone formula or algorithm is printed. Checking each run's interval against that run's estimand is a reviewer reconstruction, consistent with Table 1 ("Probability the confidence interval contains the estimand", p.841) and with the per-run procedure of Table 3 (p.845); interval-endpoint conventions are unspecified.
- **"Imbalance" distance measure, "value for money" decision rule, "robustness" set of tests:** not specified.
- **Estimand variation across draws.** "SD of estimands" (Table 1) implies the estimand can differ across runs, but the paper does not discuss how bias and RMSE treat a run-varying estimand beyond the printed `mean(estimate - estimand)`.

**Diagnosis procedure**
- **Number of simulations:** no recommendation; `m` in Table 3, 500 in the appendix. The number used for Figures 1 and 2 is not stated in the main text.
- **Bootstrap details:** what is resampled, how many replicates, and what interval is formed are not stated (p.845; footnote 11).
- **Fixed vs redrawn population:** the printed algorithm redraws everything; the paper does not discuss conditioning on a fixed population or on realized covariates, even where it mentions having "an existing data set of the units" (p.858).
- **Common random numbers across compared designs, seeds, failures:** Not discussed in paper.
- **Appendix notation.** "the diagnostic statistics, t(D, Y, f), and diagnosands, theta(D, Y, f, g)" (p.859): the arguments are not defined, `g` appears nowhere else, and `D` and `Y` here are not obviously the `D` (data strategy) of the main text.
- **Symbol reuse in Table 3.** Step 2 assigns the potential outcomes to an object named `D` (`D <- f_Y(u)`), which collides with `D` the data strategy; step 6 calls the diagnostic statistic `t` in words and `e` in code.

**Printed inconsistencies (reviewer notes; none is reconciled by the paper)**
- **Confounding illustration (p.840).** The equations use `eps_Y` and `eps_X`, but the distributional statement reads "with Z, eps_X, eps_Z distributed standard normal"; `eps_Z` is not otherwise defined and `eps_Y` is given no distribution.
- **Alternative-model notation (p.840).** The comparison under an alternative model is printed as `P_M(a^A - a^{M'} | D)`, with the prime on the estimand superscript only.
- **Observational regression example (p.849).** The closing sentence reads "even though the answer strategy 'assumes' a nonlinear relationship in M that does not hold" (the word is hyphenated across a line break in print). The surrounding text describes a linear answer strategy and a nonlinear model, so the sense appears to call for "linear"; transcribed as printed.
- **Figure 1 caption vs legend (p.850).** Line styles for the factorial and three-arm designs are swapped between the caption and the legend; see the Figure 1 notes above.
- **Estimator abbreviation.** The same estimator is written "BFE-IPW" and "IPW-BFE" (p.843; Table 2).
- **Table 3 vs appendix.** The two code listings differ (N = 200 vs 1000; `Y ~ Z + u` vs `Y ~ 0.25 * Z + u`; `model = lm_robust` only in the appendix). They are presented as separate illustrations, not as the same design.
- **Appendix assignment prose vs code.** "assigned to treatment independently with probability 0.5" (p.858) alongside `declare_assignment(m = 50)`; the argument is not defined.
- **Footnote 19 (p.854).** The simulated replication uses "187 clusters instead of 50", while the footnote refers to "those other 189 clusters" in the excluded factorial conditions; the two numbers describe different sets and neither is derived in the text.

**Material not reviewed**
- **Supplementary Materials.** The formal declarations behind the main-text worked examples are in the supplement (Section 1; Section 2; Sections 3.1-3.10; Section 4); only the two-arm example is declared in the article itself (Table 3, p.845; appendix, pp.858-859). The supplement was not reviewed, so the exact models, parameter values, sample sizes, simulation counts and numeric diagnosand values for those examples are not recorded here.
- **Replication archive and software.** The Dataverse archive (p.856) and the DeclareDesign package were not examined. Everything about the software in this review is limited to the nine declaration / diagnosis function names, the `+` operator (which the paper calls an operator, p.845), and the call signatures printed on pp.845 and 858-859.
- **Bibliography.** The reference list (pp.856-858) was read but not transcribed; citations are given in the short form the paper uses in its text.

**Numeric results**
- Almost no numeric diagnosand values are printed in the main text. The exceptions are: the Bayesian example ("40%", "33%", p.846), the QCA example ("100% of the time", p.848), the exaggeration ratios ("a factor of two", "a factor of four", "around one", p.855), and the tool census counts (Table 2, p.843). Figure 1 and Figure 2 values in this review are visual approximations.

**Not developed by the paper**

*Each item below means the paper gives no procedure, formula, or evidence for it. It does not mean the words never occur: passing mentions are noted where found.*
- Difference-in-differences, staggered adoption, panel or repeated-measures designs, event studies, parallel trends.
- Diagnosis on real data: placebo assignment, resampling of observed units or periods, injection of effects into observed outcomes.
- A few-cluster or frequentist small-sample inference procedure (the Bayesian descriptive-inference example does consider "a random sample of relatively few units", p.846); serial correlation; interference: no interference diagnosis is developed. The paper refers to a "spillover structure" (p.842) and a "no-spillovers assumption" (p.852), and its appendix states that the example satisfies SUTVA "though in general since Z is a vector, it need not" (p.858), so the framework itself does not require the own-assignment-only restriction.
- A minimum-detectable-effect or required-sample-size search algorithm, or any general procedure for searching over design parameters. Minimum detectable effects are mentioned once, by analogy ("Just as power calculators focus attention on minimum detectable effects", p.846), and the split-sample example suggests that "one can assess what an optimal division of units into training and testing data might be given different hypothesized effect sizes" (p.851).
- Size / false-positive rate and standard-error calibration as diagnosands.
- A universal numerical rule for selecting an estimator or for combining diagnosands. The paper mentions two conventions without prescribing thresholds: "the standard power target of 0.80" and that "Some researchers are unsatisfied unless the 'bias' diagnosand is exactly zero" (p.841).
- Validation of a diagnosis-selected design on held-out data.
- Any mechanism that binds a researcher to a declared design; the paper relies on declaration, registration and reconciliation reporting, and states that nothing in the framework enforces honest declaration (p.856).
- How to act on a detected deviation between ex ante and ex post declarations, and how large a deviation makes results "exploratory" (p.852).
