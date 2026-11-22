# Paper Review: How Much Should We Trust Differences-in-Differences Estimates?

**Authors:** Marianne Bertrand, Esther Duflo, Sendhil Mullainathan
**Citation:** Bertrand, M., Duflo, E., & Mullainathan, S. (2004). How Much Should We Trust Differences-in-Differences Estimates? The Quarterly Journal of Economics, 119(1), 249-275.
**DOI:** https://doi.org/10.1162/003355304772839588
**Source reviewed:** The published article, Quarterly Journal of Economics 119(1), February 2004, 27-page PDF = journal pp. 249-275 (main text pp. 249-274; references pp. 274-275; no appendix). The PDF was reviewed externally and is not committed to the repository. Reproduce by obtaining the article via the DOI above. All page citations below are JOURNAL page numbers. The working-paper version that the article cites for details ("Bertrand, Duflo, and Mullainathan [2002]"), the "unpublished appendix available from the authors" (footnote 15, p.258) and the bootstrap code "available upon request" (footnote 21, p.267) were NOT reviewed. Transcription conventions: Greek letters and hats are rendered in ASCII (`beta_hat`, `rho_hat_1`, `epsilon`), typographic quotes as straight quotes, the paper's dashes as hyphens, and the multiplication sign printed on p.259 as `*`. Table cells are transcribed exactly as printed, including the number of digits (for example `.44`, `.3975`, `.5`); the paper prints each standard error on the line below its rejection rate, and it is given here in parentheses in the same cell. Greek letters and mathematical symbols inside quoted passages are transliterated to ASCII (for example `tau`, `rho`, `+/-`, `x` for the multiplication sign).
**Review date:** 2026-06-26 (initial review, scoped to the placebo diagnostics in `diff_diff/diagnostics.py`); extended 2026-11-22 (full source audit: the placebo-law procedure, the effect-injection step, the inference methods evaluated, and all eight tables). The 2026-11-22 audit supersedes the summary tables of the initial review; the initial review's registry text for the placebo diagnostics is retained below.

> **Personal verification stamp (load-bearing items spot-checked against the source).** Beyond the multi-agent extraction, the reviewer checked directly against the source PDF: every cell of Table II (p.257), including standard errors; the placebo-law assignment sentences (uniform passage year 1985-1995, exactly half the states; pp.255-256); the effect-injection passages ("the outcome plus I_st times whichever effect we wish to simulate", p.259; "we add I_st * 0.02 to the data", p.261); footnote 14 on aggregation (p.258); the sampling statement in Table III note b (p.262); the bootstrap decision rule (p.265) and the aggregation t-statistic adjustment (p.267); and footnotes 2 and 7 (pp.250, 253-254). Every quoted passage of 40 or more characters was additionally run through a mechanical word-sequence comparison against the PDF text layer.

---

> **Scope of this paper (important).** BDM (2004) is an *inference* paper, not an
> estimator. It explicitly "assume[s] away biases in estimating the intervention's effect
> and instead focus[es] on issues relating to the *standard error* of the estimate"
> (p. 250). Its two enduring contributions to a DiD library are (a) the **placebo-law /
> randomization-inference diagnostic** that grounds diff-diff's `PlaceboTests` surface, and
> (b) the demonstration that **serial correlation** makes conventional DD standard errors
> grossly understate sampling variability - motivating serial-correlation-robust inference
> over the conventional OLS/HC1 default: unit-level cluster-robust SEs and block/percentile-t
> bootstrap when the number of groups is large, and time-series aggregation for few groups.

---

## Methodology Registry Entry

*Two registry-facing blocks follow, both sourced from this one paper. (1) `## PlaceboTests` is the review text behind the REGISTRY section of the same name, which covers the placebo diagnostics in `diff_diff/diagnostics.py`. (2) `## Placebo-law rejection-rate and power diagnosis (working title)` is prospective text for a capability the library does not have. Everything after the end-of-registry marker is audit material.*

## PlaceboTests

*Status: integrated. `docs/methodology/REGISTRY.md` has a `## PlaceboTests` section built from this block, and `METHODOLOGY_REVIEW.md` records the surface as reviewed. The REGISTRY section is the authority for what the library does; this block is the paper-side record it was derived from. Formatted to match REGISTRY.*

**Primary source:** [Bertrand, M., Duflo, E., & Mullainathan, S. (2004). How Much Should We Trust Differences-in-Differences Estimates? *The Quarterly Journal of Economics*, 119(1), 249-275.](https://doi.org/10.1162/003355304772839588) Paper review on file: `docs/methodology/papers/bertrand-duflo-mullainathan-2004-review.md`.

**Scope:** BDM (2004) introduces the **placebo-law experiment** - randomly assign a fake
treatment date and a fake set of treated groups, estimate the DD regression, and ask how
often a (necessarily spurious) "effect" is significant. Because the laws are fictitious,
the true effect is zero, so a correctly-sized 5% test should reject ~5% of the time
(p. 251). The fraction of placebo runs that reject is the rejection frequency under the chosen
experiment. With outcomes held fixed across draws, the paper says "the exact rejection rate we
should expect is not known, and may be different from 5 percent even for a correctly sized
test" (p. 258); to make the rates interpretable against nominal size, its later CPS simulations
resample whole state vectors with replacement before assigning each placebo law (p. 258).
Footnote 11
(p. 256) makes the link to **randomization inference** explicit: "if true laws were
randomly assigned, the distribution of the parameter estimates obtained using these placebo
laws could be used to form a randomization inference test of the significance of the DD
estimate [Rosenbaum 1996]." diff-diff's `PlaceboTests` operationalizes this idea as four
generic diagnostics (`fake_timing`, `fake_group`, `permutation`, `leave_one_out`).

**Key implementation requirements:**

*The placebo-law design (the empirical basis of the diagnostic):*
- **Two-step random assignment per replication (Section III, pp. 255-256):**
  1. Draw a fake treatment year from a **uniform distribution over 1985-1995** (footnote 9:
     bounded to guarantee pre- and post-intervention observations).
  2. Randomly designate **exactly half the groups (25 of 50 states)** as "affected."
  Then `I_st = 1` for units in an affected group after the fake date, else 0.
- **At least 200, typically 400 replications** per cell; the fraction with `|t| > 1.96` is
  the empirical rejection rate (Table II note).
- **Permutation / randomization-inference variant:** randomly reassign treatment over a
  fixed set of outcomes; the empirical distribution of placebo estimates is the reference
  distribution for a randomization test (footnote 11).
- **Serially-uncorrelated placebo (Table II, row 5):** to isolate serial correlation of the
  treatment indicator, set `I_st = 1` only in **ten randomly selected years (1979-1999)** for
  treatment-group states, 0 otherwise (instead of one permanent post-date switch); rejection
  collapses to **5%** (correct size), confirming the serial correlation of `I_st` is the
  driver of over-rejection.
- **Power injection:** to test power against a known alternative, replace the outcome by
  `outcome + I_st * delta` (e.g. `delta = 0.02`, p.259; the effect is uniform across states, footnote 16).

*Critical caveat on the reference distribution (footnote 12, p. 256):*
- "we are randomizing the treatment variable while keeping the set of outcomes fixed. In
  general, the distribution of the test statistic induced by such randomization is **not a
  standard normal distribution** and, therefore, the exact rejection rate we should expect
  is not known." The paper's own response is the resampled-states Monte Carlo study (p. 258);
  it gives no permutation p-value formula. **Reviewer inference for the diagnostic (not stated
  by BDM):** comparing a permutation estimate to normal critical values is not generally
  valid, so a permutation p-value is read off the empirical placebo distribution (proportion
  of placebo `|estimate|` at least as large as observed), not from a t/normal table. The
  particular tail statistic and finite-draw calculation are library choices; see REGISTRY
  `## PlaceboTests`.

*Estimator equation under test:* equation (1) (p.250) and its state-year aggregated analogue (footnote 14, p.258); both are transcribed in the next block.

*Edge cases:*
- NaN inference for undefined statistics (diff-diff defensive convention; not from BDM):
  - `permutation_test`: t_stat is NaN when permutation SE is zero (all permutations produce identical estimates).
  - `leave_one_out_test`: t_stat, p_value, CI are NaN when LOO SE is zero (all LOO effects identical).
  - **Note:** Defensive enhancement matching CallawaySantAnna NaN convention.

**Reference implementation(s):**
- No canonical R/Stata package implements BDM's *placebo battery* as a single command; the
  paper's own machinery is custom (block-bootstrap "codes... available upon request",
  footnote 21). BDM's two inference *corrections* that did become standard tooling are:
  - Stata `cluster` (arbitrary/state-level cluster-robust VCV, Section IV.E).
  - Stata `xtgls` (parametric AR(k) corrections, Section IV.A - shown *not* to work).

**Requirements checklist:**

*Directly grounded in BDM (2004):*
- [ ] Placebo-law size experiment - randomly assign a fake treatment date and a fake treated group, estimate the DD, and measure the rejection rate (about 5% under the null for a correctly sized test when states are resampled; with outcomes held fixed the expected rate is not known, p. 258) (Section III; `permutation` / `fake_group` operationalize the random-assignment exercise).
- [ ] BDM note, conditionally, that placebo estimates "could be used to form a randomization inference test" if true laws were randomly assigned (footnote 11), and that the test statistic under fixed-outcome randomization is not standard normal (footnote 12). The paper gives no p-value formula.
- [ ] Power can be probed by injecting a known additive effect `outcome + I_st * delta` (BDM's `delta = 0.02` exercise, p. 261).
- [ ] The inference lesson holds - serial-correlation-robust SEs (unit-level cluster / block bootstrap for large N; aggregation for small N) rather than OLS/HC1 (Section IV, Conclusion).

*Library extensions of the placebo idea (not prescribed by BDM):*
- [ ] `placebo_timing_test` against a *chosen* pre-treatment period, reframed as a pre-trends diagnostic (BDM draw the fake date at random for a size experiment, not to test a specific pre-period).
- [ ] `leave_one_out` single-unit sensitivity diagnostic.
- [ ] Permutation p-value read from the empirical placebo distribution of absolute estimates rather than from normal/t critical values (reviewer inference from footnotes 11-12; the tail statistic is not specified by BDM).
- [ ] Permutation p-value floor at `1/(n_valid+1)` (finite-draw randomization-inference convention; not stated by BDM).

## Placebo-law rejection-rate and power diagnosis (working title)

**Status: proposed text for a future REGISTRY section; this block is a non-authoritative source audit.** REGISTRY's existing `## PlaceboTests` section (previous block) covers single placebo fits, a permutation test and a leave-one-out check; it has no section for repeating a placebo assignment many times to report a rejection rate, with or without an injected effect, which is what this block concerns. The paper is the candidate source for the PLACEBO-LAW PROCEDURE WITH REJECTION RATES AND POWER ON REAL DATA of a planned "design diagnosis" capability (assign placebo treatments on a user's own untreated data, optionally inject a known effect, and report rejection rate and power before any real analysis). It is reviewed as a set with the companion reviews of Coglianese and Fang (2026) and Blair, Cooper, Coppock and Humphreys (2019). Nothing below is a methodology contract until a REGISTRY section is written and reviewed alongside an implementation.

*This is an INFERENCE-EVALUATION paper, not an estimator paper: it proposes no new treatment-effect estimator, states no theorem, and prints exactly one numbered equation (the DD regression, equation (1)). The standard registry headings (estimator equation, standard errors) therefore do not apply and are replaced by the placebo-law procedure, the effect-injection step, the two reported metrics, the simulated-data designs, and the roster of inference methods evaluated. The registry-candidate text ends at the marker before `## Results`; everything below that marker is audit material (transcribed tables, quotes, implementation notes, gaps) and is NOT part of the proposed registry text.*

*The heading above is provisional. The paper's own terms are "placebo laws" and "placebo interventions", evaluated by "rejection rates" under "No effect" and under a "2% effect"; the name of any library surface is to be decided in the implementation plan.*

**Primary source:** [Bertrand, M., Duflo, E., & Mullainathan, S. (2004). How Much Should We Trust Differences-in-Differences Estimates? The Quarterly Journal of Economics, 119(1), 249-275.](https://doi.org/10.1162/003355304772839588)

**Key implementation requirements:**

*Scope of the source:*
- What is studied is the STANDARD ERROR of the DD estimate, not the point estimate: "In this paper we address an altogether different problem with DD estimation. We assume away biases in estimating the intervention's effect and instead focus on issues relating to the standard error of the estimate." (p.250)
- OLS on equation (1), a regression with state and year fixed effects and a single intervention dummy (p.250), is the baseline estimator throughout. Two qualifications: the aggregation methods re-estimate the effect by OLS on a collapsed two-period panel (p.267), and the AR(k) corrections are run with a GLS command (`xtgls`, Table IV note e) without the paper saying whether the reported coefficient is the OLS or the GLS one. What varies across the paper's exhibits is how the standard error or the test is formed, the data, the number of states `N`, the number of periods `T`, and the serial correlation of the outcome and of the intervention variable.
- Setting evaluated: one data set - women's log weekly earnings in the Current Population Survey, 50 states by 21 years (p.255) - plus simulated AR-type data calibrated to it (p.260). Three other CPS outcomes appear in one table only (Table II rows 6-8, p.257).
- Two quantities are reported for every design: the rejection rate of the null of no effect under placebo laws ("No effect"), and the rejection rate after a known 2 percent effect is added to the placebo-treated observations ("2% effect"). These are the only two simulation PERFORMANCE quantities. The paper also tabulates estimated autocorrelation coefficients (Table II; an "Estimated rho_hat_1" column in Table IV) and the survey statistics of Table I - see "What the paper does NOT provide".

*Rationale (pp.250-252, 256):*
- "In this paper we argue that the estimation of equation (1) is in practice subject to a possibly severe serial correlation problem. While serial correlation is well understood, it has been largely ignored by researchers using DD estimation. Three factors make serial correlation an especially important issue in the DD context. First, DD estimation usually relies on fairly long time series. Our survey of DD papers, which we discuss below, finds an average of 16.5 periods. Second, the most commonly used dependent variables in DD estimation are typically highly positively serially correlated. Third, and an intrinsic aspect of the DD model, the treatment variable I_st changes itself very little within a state over time. These three factors reinforce each other so that the standard error for beta_hat could severely understate the standard deviation of beta_hat." (p.251)
- The logic of the placebo exercise: "To assess the extent of this problem, we examine how DD performs on placebo laws, where treated states and year of passage are chosen at random. Since these laws are fictitious, a significant 'effect' at the 5 percent level should be found roughly 5 percent of the time." (p.251)
- "This is analogous to asking 'If hundreds of researchers analyzed the effects of various laws in the CPS, what fraction would find a significant effect even when the laws have no effect?'" (p.256)

*The DD regression (equation (1), p.250):*

```
(1)    Y_ist = A_s + B_t + c X_ist + beta I_st + epsilon_ist
```

"where A_s and B_t are fixed effects for states and years, respectively, X_ist are relevant individual controls and epsilon_ist is an error term. The estimated impact of the intervention is then the OLS estimate beta_hat." (p.250). `Y_ist` is "the outcome of interest for individual i in group s (such as a state) by time t (such as a year)" and `I_st` is "a dummy for whether the intervention has affected group s at time t" (p.250).

The state-year aggregated analogue (footnote 14, p.258; unnumbered in the paper):

```
Y_bar_st = alpha_s + gamma_t + beta I_st + epsilon_st
```

where `Y_bar_st` is the state-year mean of the residuals from a first-stage regression of individual log weekly earnings on the individual controls (see step 8 below).

*Placebo-law procedure (Section III, pp.254-259):*

The paper prints no numbered algorithm. The steps below are distilled from the prose; each step cites the page it comes from.

1. **Data source and sample.** "We extract data on women in their fourth interview month in the Merged Outgoing Rotation Group of the CPS for the years 1979 to 1999. We focus on all women between the ages 25 and 50. We extract information on weekly earnings, employment status, education, age, and state of residence. The sample contains nearly 900,000 observations." (p.255)
2. **Outcome.** "We define wage as log(weekly earnings). Of the 900,000 women in the original sample, approximately 540,000 report strictly positive weekly earnings." (p.255)
3. **Unit and periods.** State by year: "This generates (50 * 21 = 1050) state-year cells, with each cell containing on average a little more than 500 women with strictly positive earnings." (p.255). 50 states, 21 years (1979-1999).
4. **Randomize the passage year.** "We first draw a year at random from a uniform distribution between 1985 and 1995." (pp.255-256). Footnote 9 (p.256): "We choose to limit the intervention date to the 1985-1995 period to ensure having enough observations prior and post-intervention." One common passage year per placebo law.
5. **Randomize which states are treated.** "Second, we select exactly half the states (25) at random and designate them as 'affected' by the law." (p.256). The number treated is fixed at half; only the identity of the treated states is random. Reviewer reading: selecting "exactly half the states" implies 25 distinct states, that is, selection without replacement; the paper does not use those words.
6. **Define the intervention variable.** "The intervention variable I_st is then defined as a dummy variable which equals 1 for all women that live in an affected state after the intervention date, 0 otherwise." (p.256). The intervention is absorbing: once on, it stays on.
7. **Estimate equation (1) by OLS and record two numbers.** "We can then estimate equation (1) using OLS on these placebo laws. The estimation generates an estimate of the law's 'effect' and a standard error for this estimate." (p.256). In the micro-data version, "The control variables X_ist include four education dummies (less than high school, high school, some college, and college or more) and a quartic in age as controls." (p.256). All regressions include state and year fixed effects (Table II note c, p.257).
8. **Aggregation of the micro data (used from Table II row 3 onward).** "In row 3 we aggregate the data into state-year cells to construct a panel of 50 states over 21 years and then estimate the analogue of equation (1) on this data." (p.258). Footnote 14 (p.258): "To aggregate, we first regress individual log weekly earnings on the individual controls (education and age) and form residuals. We then compute means of these residuals by state and year: Y_bar_st. On these aggregated data, we estimate Y_bar_st = alpha_s + gamma_t + beta I_st + epsilon_st. The results do not change if we also allow for heteroskedasticity when estimating this equation."
9. **Repeat with new laws on the same data.** "To understand how well conventional DD performs, we can repeat this exercise a large number of times, each time drawing new laws at random." (p.256). "For each of these simulations we randomly generate new laws but use the same CPS data." (p.256). Number of replications: "We estimate equation (1) for at least 200 independent draws of placebo laws." (p.256); table notes: "The number of simulations for each cell is at least 200 and typically 400." (Table II note a, p.257; "typically 400 and at least 200" in Tables III-VIII).
10. **Compute the rejection rate.** "If OLS were to provide consistent standard errors, we would expect to reject the null hypothesis of no effect (beta = 0) roughly 5 percent of the time when we use a threshold of 1.96 for the absolute t-statistic." (p.256). "We report the fraction of simulations in which the absolute value of the t-statistic was greater than 1.96." (p.256). The test is two-sided (absolute t-statistic) at the 5 percent significance level (table notes), with a fixed normal critical value of 1.96.
11. **Resampled-states variant (the paper's default from Table II row 4 onward).** With laws randomized over a fixed set of outcomes, "the exact rejection rate we should expect is not known, and may be different from 5 percent even for a correctly sized test" (p.258; also footnote 12, p.256). The remedy: "we perform a Monte Carlo study where the data generating process is the state-level empirical distribution of the CPS data. Specifically, for each new simulation, we sample states with replacement from the CPS, putting probability 1/50 on each of the 50 states. Because we sample entire state vectors, this preserves the within-state autocorrelation of outcomes. In each sample, we then randomly pick half of the states to be 'treated' and randomly pick a treatment year (as explained above)." (p.258). "To facilitate the interpretation of the rejection rates, all the CPS results presented below are based on such Monte Carlo simulations using the state-level empirical distribution of the CPS data." (pp.258-259)
12. **Optional: inject a known effect and repeat** (next block).

Reviewer reconstruction (NOT printed in the paper) of the reported quantity, with `R` the number of placebo draws:

```
RejRate = (1/R) * sum_{r=1}^{R} 1( | beta_hat_r / SE(beta_hat_r) | > 1.96 )      [conventional OLS and other normal-critical-value rows]
RejRate = (1/R) * sum_{r=1}^{R} 1( method's own decision rule rejects in replication r )      [e.g. block bootstrap: reject "if 95 percent of the t_r are smaller than t", p.265; aggregation: t-statistic "adjusted to take into account the smaller number of observations", p.267]
```

Each table also reports a standard error of the rejection rate, described only as "computed using the number of simulations" (table notes d, Tables III-VIII). Reviewer reconstruction: many printed values are consistent with `sqrt( RejRate * (1 - RejRate) / R )` for `R` between about 200 and 400 (for example .49 with (.025) at `R = 400`), but not all: Table IV row 6 prints .06 with (.023), which under this formula implies `R` of about 107, below the stated minimum of 200. The formula itself is not printed.

Summary of what is randomized in each design the paper uses:

| Design | Outcome data | States treated | Law date | Where used |
|---|---|---|---|---|
| Fixed-outcome placebo laws | Observed CPS data, unchanged across draws | Exactly half (25 of 50), drawn at random | One year, uniform on 1985-1995, common to all treated states | Table II rows 1-3 (pp.256-258) |
| Resampled-states Monte Carlo | States drawn with replacement, probability 1/50 each, entire state vectors | Half of the sampled states, drawn at random | One year drawn "as explained above" | Table II rows 4, 6-8; all "CPS" rows of Tables III-VIII (pp.258-259) |
| Serially uncorrelated laws | Resampled-states | Half, drawn at random | Ten dates drawn at random between 1979 and 1999; law equals 1 at those dates only | Table II row 5 (p.259) |
| Staggered laws | Resampled-states (and AR(1) data) | Half, drawn at random | A separate passage date per treated state, "uniformly drawn between 1985 and 1995" | Table VI rows 4, 8, 12, 16, 19 (footnote 22, p.267) |
| Simulated AR(1) | New simulated data set for every draw | Placebo laws generated for each simulation | Not restated for simulated data | Table II Panel B; Panel B of Tables III-VIII (p.260) |

The serially uncorrelated intervention, verbatim: "As before, we randomly select half of the states to form the treatment group. However, instead of randomly choosing one date after which all the states in the treatment group are affected by the law, we randomly select ten dates between 1979 and 1999. The law is now defined as 1 if the observation relates to a state that belongs to the treatment group at one of these ten dates, 0 otherwise. In other words, the intervention variable is now repeatedly turned on and off, with its value in one year telling us nothing about its value the next year." (p.259)

*Effect injection and power (Section III, p.259; Section IV, p.261):*

The paper DOES add a non-zero effect to the placebo-treated observations. The defining sentences, verbatim:

> "We have so far focused on Type I error. A small variant of the exercise above allows us to assess Type II error, or power against a specific alternative. After constructing the placebo intervention, I_st, we can replace the outcome in the CPS data by the outcome plus I_st times whichever effect we wish to simulate. For example, we can replace log(weekly earnings) by log(weekly earnings) plus I_st * .0x to generate a true .0x log point (approximately x percent) effect of the intervention. By repeatedly estimating DD in this altered data (with new laws randomly drawn each time) and counting rejections, we can assess how often DD rejects the null of no effect under a specific alternative. Under the alternative of a 2 percent effect, OLS rejects the null of no effect in 66 percent of the simulations (row 4, last column)." (p.259)

> "It is important to note that the 'effect' we generate is uniform across states. For some practical applications, one might also be interested in cases where the treatment effect is heterogeneous." (footnote 16, p.259)

> "We also evaluate the power of each estimator against the specific alternative of a 2 percent effect (we add I_st * 0.02 to the data). The choice of 2 percent as the alternative is admittedly somewhat arbitrary, but our conclusions on the relative power of each estimator do not depend on this specific value." (p.261)

> "We report the power against the alternative of 2 percent because 2 percent appears as a 'reasonable' size effect. Moreover, in simulated data with an AR(1) process with rho = 0.8, the rejection rate when using the true variance-covariance matrix is 32.5 percent when there is a 2 percent effect, which is large enough to be very different from the 5 percent rejection rate obtained under the null of no effect." (footnote 18, p.261)

> "The data used in the last column were altered to simulate a true 2 percent effect of the intervention." (note a of Tables II-VIII)

What these sentences pin down:
- **Additive on the outcome scale.** The outcome is replaced by "the outcome plus I_st times" the effect. For the log-wage outcome the added constant is 0.02 log points, described as "approximately" 2 percent. It is additive in logs, hence approximately proportional in levels.
- **Applied to the placebo-treated observations only.** The added term is `I_st * 0.02`, so it is non-zero exactly where `I_st = 1` (treated states, after the intervention date) and zero elsewhere.
- **Constant.** One value for every treated state ("uniform across states", footnote 16) and, since the term is `I_st * 0.02`, for every period in which `I_st = 1`. No dynamic, ramping or heterogeneous effect is simulated.
- **Laws are redrawn in every replication** ("with new laws randomly drawn each time", p.259).
- **A single effect size.** Only the 2 percent alternative is tabulated. The general ".0x" wording allows any value; no power curve over effect sizes is reported.
- **How power is reported.** As the fraction of replications in which the null `beta = 0` is rejected, using the same test as in the no-effect column, printed in the "2% effect" column of Tables II-VIII with a standard error.
- **No size adjustment.** The "2% effect" column is the raw rejection frequency of each test at its own (possibly distorted) size. For a test that overrejects under no effect, the paper does not report a size-corrected power figure. Reviewer note: for conventional OLS the 66.3 percent "power" on CPS data therefore sits alongside a 49 percent rejection rate under no effect (Table II row 4).
- **Benchmark for power.** The paper treats the rejection rate obtained "with the correct covariance matrix" as an upper bound: "the rejection rate in the case of a 2 percent effect was 78 percent in a Monte Carlo simulation with no autocorrelation and 32 percent in AR(1) data with rho = .8" (p.271), citing Tables II and IV; footnote 18 gives 32.5 percent for the latter.

*Monte Carlo design on simulated data (Section III, p.260; Table II Panel B; Table IV Panel B):*

- Motivation: "one might be concerned that we are by chance detecting actual laws or other relatively discrete changes. Also, there might be other features of the CPS wage data, such as state-specific time trends, that may also give rise to overrejection. To address this issue, we replicate our analysis in an alternative Monte Carlo study where the data generating process is an AR(1) model with normal disturbances." (p.260)
- Calibration: "The data are generated so that their variance structure in terms of relative contribution of state and year fixed effects matches the empirical variance decomposition of female state wages in the CPS." (p.260). Table notes: "an AR(1) model with normal disturbances chosen to match the CPS state female wage variances" (Table II note b, p.257). No variance values are printed.
- Redrawn every replication: "We randomly generate a new data set and placebo laws for each simulation. By construction, we can now be sure that there are no ambient trends and that the laws truly have no effect." (p.260)
- AR(1) parameters used: `rho = .8` (the main case; Table II row 9 and Panel B of every later table), and `rho = 0, .2, .4, .6, -.4` (Table II rows 10-14). Rationale for .8 is in footnote 8 (p.255): the estimated first-order coefficient of 0.51 "would correspond to a true autocorrelation coefficient of 0.6 if the data generating process were an AR(1)", but "To match the estimated second- and third-order autocorrelation parameters, the data would have to follow an AR(1) process with an autocorrelation coefficient of 0.8."
- Disclaimer: "We choose an AR(1) process to illustrate the problems caused by autocorrelation in the context of a simple example, not because we think that such a process matches the female wage data the best." (footnote 17, p.260)
- Two further data generating processes are used only to stress the parametric correction (Table IV rows 7-8, p.263; text p.264): an AR(2) with `rho_1 = .55` and `rho_2 = .35` ("These parameters were chosen because they match well the estimated first, second, and third autocorrelation parameters in the wage data when we apply the formulas to correct for small sample bias given in Solon [1984]"), and "the sum of an AR(1) (with autocorrelation parameter 0.95) plus white noise (the variance of the white noise is 13 percent of the total variance of the residual)".
- Sample sizes: 50 states and 21 periods by default; Table III varies `N` over 50, 20, 10, 6 and `T` over 21, 11, 5, 3 and 50 for the AR(1) data (and `T` over 21, 11, 7, 5, 3, 2 for the CPS-based data). "N refers to the number of states used in the simulation and T refers to the number of years." (Table III note e, p.262)
- The first Monte Carlo study (resampled CPS states) and the second (AR(1)) are distinguished in footnote 4 (p.251): "In the first Monte Carlo study, the data generating process is the state-level empirical distribution that puts probability 1/50 on each of the 50 states' observations in the CPS. As the randomization is at the state level, this preserves the within-state autocorrelation structure. In the second Monte Carlo study, the data generating process is an AR(1) with normal disturbances."

*Serial-correlation diagnostics the paper computes on the real data (p.255):*

"We estimate first, second, and third autocorrelation coefficients for the mean state-year residuals from a regression of wages on state and year dummies (the relevant residuals since DD includes these dummies). The autocorrelation coefficients are obtained by a simple OLS regression of the residuals on the corresponding lagged residuals. We are therefore imposing common autocorrelation parameters for all states. The estimated first-order autocorrelation coefficient is 0.51, and is strongly significant. The second- and third-order autocorrelation coefficients are high (0.44 and 0.33, respectively) and statistically significant as well. They decline much less rapidly than one would expect if the data generating process was a simple AR(1)." (p.255). Table II reports the three coefficients for each CPS outcome.

*Inference methods evaluated (Sections III-IV, pp.256-272):*

The paper calls these "alternative estimators" and "alternative estimation techniques" (pp.251-252, 261); they are alternative ways of forming the standard error or the test for the same DD coefficient. Only the cluster-robust formula is displayed as an equation, and it is unnumbered. Rates below are "no effect / 2% effect" as printed; full tables are in the Results section.

| Method | Section, pages | How the standard error or test is formed | N = 50 result (CPS-based unless noted) | Smaller N | Authors' verdict |
|---|---|---|---|---|---|
| Conventional OLS, micro data | III, p.256; Table II row 1 | OLS standard error on equation (1), "without any correction for grouped error terms" | .675 / .855 (fixed outcomes) | - | "a stunning 67.5 percent" (p.256) |
| OLS, clustered at the state-year level | III, p.258; Table II row 2 | "an arbitrary correlation of the error terms at the state-year level"; "implemented by using the 'cluster' command in STATA" (footnote 13) | .44 / .74 (fixed outcomes) | - | "We still find a very high (44 percent) rejection rate." (p.258) |
| OLS on state-year aggregates | III, p.258; Table II rows 3-4 | OLS standard error on the aggregated analogue of equation (1) (footnote 14) | .435 / .72 (fixed outcomes); .49 / .663 (resampled states) | 20: .39 / .54; 10: .443 / .510; 6: .383 / .433 (Table III) | "correlated shocks within state-year cells explain only part of the overrejection" (p.258) |
| Parametric AR(1) correction | IV.A, pp.263-265; Table IV | Estimate an AR(1) for the residual and use it to form the block-diagonal variance-covariance matrix | .24 / .66 (estimated `rho_hat_1` = .381); .18 / .363 (imposing `rho` = .8) | Not reported | "do not appear to provide an easy solution for the applied researcher" (pp.264-265) |
| Block bootstrap | IV.B, pp.265-267; Table V | Bootstrap the absolute t-statistic, resampling whole states | .065 / .26; AR(1) data: .05 / .25 | 20: .13 / .19; 10: .225 / .25; 6: .435 / .375 | "a reliable solution to the serial correlation problem when the number of groups is large enough" (p.267) |
| Ignoring time-series information: simple aggregation | IV.C, pp.267-269; Table VI | Average the data before and after the law; run equation (1) on a panel of length 2 | .053 / .163; AR(1) data: .050 / .243 | 20: .050 / .088; 10: .053 / .065; 6: .068 / .07 | Works "even for quite small numbers of states"; "power is quite low and diminishes fast with sample size" (pp.267, 269) |
| Ignoring time-series information: residual aggregation | IV.C, pp.267-269; Table VI | Partial out state effects, year dummies and covariates; collapse treated-state residuals into before and after; OLS on the two-period panel | .058 / .173; staggered laws: .048 / .363; AR(1) data: .045 / .235; AR(1) staggered: .075 / .355 | 20: .06 / .183; 10: .093 / .178; 6: .11 / .123 (staggered: .048 / .130; .088 / .128; .09 / .138) | "performs a little worse, but the overrejection remains relatively small" (p.267) |
| Empirical variance-covariance matrix | IV.D, pp.269-270; Table VII | Estimate a common T by T within-state covariance block from the cross-section of states | .055 / .243; AR(1) data: .07 / .25 | 20: .08 / .138; 10: .105 / .145; 6: .153 / .185 | "performs well when the number of states is large (N = 50)"; "performs more poorly for small sample sizes" (p.269) |
| Arbitrary variance-covariance matrix (cluster at the state level) | IV.E, pp.270-272; Table VIII | Generalized White-like formula with whole states as clusters | .063 / .268; AR(1) `rho` = .8: .045 / .275; AR(1) `rho` = 0: .035 / .74 | 20: .058 / .13; 10: .08 / .12; 6: .115 / .118 | "Despite its generality, the arbitrary variance-covariance matrix does quite well." (p.271) |

Method details as printed:

- **Conventional OLS and the grouped-error issue (pp.250, 256-258).** "Standard errors used to form confidence interval for beta_hat are usually OLS standard errors, sometimes corrected to account for the correlation of shocks within each state-year cell." (p.250). "OLS assumes that the variance-covariance matrix for the error term is diagonal while in practice it might be block diagonal, with correlation of the error terms within each state-year cell." (pp.256, 258). Footnote 13 (p.258) adds that the Moulton correction "forces a constant correlation of the error terms at the state-year level" and that "The rate of rejection of the null hypothesis of no effect was not statistically different under the Moulton technique."

- **Parametric AR(1) correction (Section IV.A, pp.263-264).** "A first possible solution to the serial correlation problem would be to specify an autocorrelation structure for the error term, estimate its parameters, and use these parameters to compute standard errors." (p.263). "Row 2 performs the simplest of these parametric corrections, wherein an AR(1) process is estimated in the data, without correction for small sample bias in the estimation of the AR(1) parameter. We first estimate the first-order autocorrelation coefficient of the residual by regressing the residual on its lag, and then use this estimated coefficient to form an estimate of the block-diagonal variance-covariance matrix of the residual." (p.264). "The results are the same whether or not we assume that each state has its own autocorrelation parameter." (p.264). Implementation: "The AR(k) corrections are implemented in stata using the 'xtgls' command." (Table IV note e, p.263). No formula is printed. Two stated reasons for failure: (i) "the downward bias in the estimator of the autocorrelation coefficient. As is already well understood, with short time-series, the OLS estimation of the autocorrelation parameter is biased downwards. In the CPS data, OLS estimates a first-order autocorrelation coefficient of only 0.4. Similarly, in the AR(1) model where we know that the autocorrelation parameter is .8, a rho_hat of .62 is estimated (row 5)." (p.264); (ii) "we have not correctly specified the autocorrelation process. As noted earlier, an AR(1) does not fit the correlogram of wages in the CPS." (p.264). Conclusion: "Any misspecification of the data generating process results in inconsistent standard errors and, at least without much deeper exploration into specification tests, it is difficult to find the true data generating process." (p.265)

- **Block bootstrap (Section IV.B, pp.265-267).** Verbatim: "Block bootstrap [Efron and Tibshirani 1994] is a variant of bootstrap which maintains the autocorrelation structure by keeping all the observations that belong to the same group (e.g., state) together. In practice, we bootstrap the t-statistic as follows. For each placebo intervention we compute the absolute t-statistic t = abs(beta_hat/SE(beta_hat)), using the OLS estimate of beta and its standard error. We then construct a bootstrap sample by drawing with replacement 50 matrices (Y_bar_s, V_s), where Y_bar_s is the entire time series of observations for state s, and V_s is the matrix of state dummies, time dummies, and treatment dummy for state s. We then run OLS on this sample, obtain an estimate beta_hat_r and construct the absolute t-statistic t_r = abs(beta_hat_r - beta_hat)/SE(beta_hat_r). The sampling distribution of t_r is random and changing as N (the number of states) grows. The difference between this distribution and the sampling distribution of t becomes small as N goes to infinity, even in the presence of arbitrary autocorrelation within states and heteroskedasticity. We draw a large number (200) of bootstrap samples, and reject the hypothesis that beta = 0 at a 95 percent confidence level if 95 percent of the t_r are smaller than t." (p.265)

  ```
  Original sample:     t   = abs( beta_hat / SE(beta_hat) )
  Bootstrap sample r:  draw with replacement 50 matrices (Y_bar_s, V_s); run OLS
                       t_r = abs( beta_hat_r - beta_hat ) / SE(beta_hat_r)
  Decision:            reject beta = 0 if 95 percent of the t_r are smaller than t
  ```

  Resampling unit: the whole state (its entire outcome time series together with its own regressor rows, including its treatment dummy path). Statistic bootstrapped: the absolute t-statistic, recentered at the original `beta_hat`. Critical value: the 95th percentile of the bootstrap distribution of `t_r`, used in place of 1.96. Number of bootstrap draws: 200 in the text (p.265) but "The bootstraps involve 400 repetitions" in Table V note a (p.266) - both as printed.

- **Ignoring time-series information (Section IV.C, pp.267-269).** Simple aggregation: "To do this, one could simply average the data before and after the law and run equation (1) on this averaged outcome variable in a panel of length 2." (p.267). Its limitation: "Taken literally, however, this solution will work only for laws that are passed at the same time for all the treated states. If laws are passed at different times, 'before' and 'after' are no longer the same for each treated state and not even defined for the control states." (p.267). Residual aggregation for staggered dates: "First, one can regress Y_st on state fixed effects, year dummies, and any relevant covariates. One can then divide the residuals of the treatment states only into two groups: residuals from years before the laws, and residuals from years after the laws. The estimate of the laws' effect and its standard error can then be obtained from an OLS regression in this two-period panel." (p.267). Small-sample adjustment: "When the number of states is small, the t-statistic needs to be adjusted to take into account the smaller number of observations (see Donald and Lang [2001] for a discussion of inference in small-sample aggregated data set). When we do that, simple aggregation continues to perform well, even for quite small numbers of states." (p.267). The adjustment itself (reference distribution, degrees of freedom) is not printed. Staggered placebo laws for this table: "we randomly choose half of the states to form the treatment group and randomly choose a passage date (uniformly drawn between 1985 and 1995) separately for each state in the treatment group." (footnote 22, p.267)

- **Empirical variance-covariance matrix (Section IV.D, pp.269-270).** "Specifically, suppose that the autocorrelation process is the same across all states and that there is no cross-sectional heteroskedasticity. In this case, if the data are sorted by states and (by decreasing order of) years, the variance-covariance matrix of the error term is block diagonal, with 50 identical blocks of size T by T (where T is the number of time periods). Each of these blocks is symmetric, and the element (i, i + j) is the correlation between epsilon_i and epsilon_{i-j}. We can therefore use the variation across the 50 states to estimate each element of this matrix, and use this estimated matrix to compute standard errors. Under the assumption that there is no heteroskedasticity, this method will produce consistent estimates of the standard error as N (the number of groups) goes to infinity [Kiefer 1980]." (p.269). No formula is printed for the estimated block or for the resulting standard error. Stated limitation: "it is only consistent under the assumption of cross-sectional homoskedasticity, an assumption that is likely to be violated in practice for many data sets" (p.270).

- **Arbitrary variance-covariance matrix (Section IV.E, pp.270-272).** "this method can be generalized to an estimator of the variance-covariance matrix which is consistent in the presence of any correlation pattern within states over time. Of course, we cannot consistently estimate each element of the variance-covariance matrix in this case, but we can use a generalized White-like formula to compute the standard errors [White 1984; Arellano 1987; Kezdi 2002]." (pp.270-271). The two displayed equations (p.271, unnumbered):

  ```
  W = (V'V)^{-1} ( sum_{j=1}^{N} u_j' u_j ) (V'V)^{-1}

  u_j = sum_{t=1}^{T} e_jt v_jt
  ```

  "where N is the total number of states, V is matrix of independent variables (year dummies, state dummies and treatment dummy), and u_j is defined for each state to be" the second line, "where e_jt is the estimated residual for state i at time t and v_jt is a row vector of dependent variables (including the constant)." (p.271; "state i" and "dependent variables" are as printed). Footnote 23: "This is analogous to applying the Newey-West correction [Newey and West 1987] in the panel context where we allow for all lags to be potentially important." Footnote 24: "This is implemented in a straightforward way by using the cluster command in STATA and choosing entire states (and not only state-year cells) as clusters." Consistency: "This estimator of the variance-covariance matrix is consistent for fixed panel length as the number of states tends to infinity [Kezdi 2002]." (p.271). Footnote 25: "Note that the resulting variance-covariance matrix is of rank TN - N. The standard error of the coefficient of the state dummies is not identified in this model. However the other terms of the variance-covariance matrix are identified and consistently estimated as N goes to infinity." No finite-sample (degrees-of-freedom) multiplier is printed.

- **Randomization inference.** Appears only as a remark and is not evaluated: "if true laws were randomly assigned, the distribution of the parameter estimates obtained using these placebo laws could be used to form a randomization inference test of the significance of the DD estimate [Rosenbaum 1996]." (footnote 11, p.256). No randomization-inference rejection rate is reported in any table.

*Scope conditions (as stated or as built into the designs):*
- **Number of states.** 50 in the base design; 20, 10 and 6 in Tables III and V-VIII. The asymptotic methods are justified "as N (the number of groups) goes to infinity" (p.269) or "as the number of states tends to infinity" (p.271), and all three (block bootstrap, empirical, arbitrary) overreject at 10 and 6 states.
- **Number of treated states.** Always half ("exactly half the states (25)", p.256; "randomly pick half of the states", p.258; footnote 22). The share treated is never varied in the published tables. Footnote 10 (p.256) says that "changing the number of 'affected' states" was tried with "similar effects" and refers to the working-paper version for details.
- **Law dates.** A single common date in the base design. Staggered dates appear only in Table VI (residual aggregation) and in the footnote 10 remark; conventional OLS, block bootstrap, the empirical matrix and the arbitrary matrix are not tabulated under staggered laws.
- **Number of periods.** 21 in the base design; Table III varies it for conventional OLS only.
- **Serial correlation of the outcome.** Positive and persistent in the base design (`rho_hat` = .509, .440, .332). "the overrejection problem diminishes with the serial correlation in the dependent variable" (p.259); with a negative first-order autocorrelation "OLS lead us to reject the null of no effect in less than 5 percent of the simulations" (p.260).
- **Serial correlation of the treatment variable.** "if the intervention variable were not serially correlated, OLS standard errors should be consistent" (p.259); Table II row 5 gives 5 percent for the on-and-off law.
- **Effect homogeneity.** The injected effect "is uniform across states" (footnote 16, p.259).
- **Outcome.** Among the CPS-based results, all except Table II rows 6-8 are for log weekly earnings of women aged 25-50. The simulated-data panels use generated AR(1) outcomes calibrated to the CPS wage variances (p.260).

*Edge cases:*
- **Fixed outcomes vs resampled outcomes.** With laws randomized over a fixed set of outcomes the correct rejection rate is unknown: "Note that we are randomizing the treatment variable while keeping the set of outcomes fixed. In general, the distribution of the test statistic induced by such randomization is not a standard normal distribution and, therefore, the exact rejection rate we should expect is not known." (footnote 12, p.256). The authors respond by resampling states with replacement (step 11) and report that the two exercises give "very similar" results (p.258): .435 (fixed, Table II row 3) vs .49 (resampled, row 4).
- **Real laws and trends in the real data.** "one might be concerned that we are by chance detecting actual laws or other relatively discrete changes. Also, there might be other features of the CPS wage data, such as state-specific time trends, that may also give rise to overrejection." (p.260). The paper's response is the simulated AR(1) study; it applies no screen for real policy events to the CPS data.
- **Passage-year window.** The placebo passage year is restricted to 1985-1995 within 1979-1999 "to ensure having enough observations prior and post-intervention" (footnote 9, p.256).
- **Sensitivity to the passage year.** "We have also run simulations where we fix the treatment year across all simulations (unpublished appendix available from the authors). The rejection rates do not vary much from year to year, and remain above 30 percent in every single year." (footnote 15, p.258)
- **Negative serial correlation gives underrejection.** Table II row 8 (changes in log wage; `rho_hat_1` = -.046) prints a no-effect rejection rate of 0, and row 14 (AR(1), `rho` = -.4) prints .008. "When rho is negative (row 14), there is underrejection." (p.260)
- **Sampling error in cell means.** "The small sample sizes in each state-year cell can lead to large sampling error and lower serial correlation in the CPS than in other administrative data. [...] Sampling error may also contribute to complicating the autocorrelation process, making it, for example, a combination of AR(1) and white noise." (footnote 8, p.255)
- **Micro data vs aggregates.** The authors "assume that the researchers estimating equation (1) have already accounted for this problem, either by allowing for appropriate random group effects or, as we do, by collapsing the data to a higher level of aggregation (such as state-year cells)" (footnote 3, pp.250-251).
- **Diluted treatment.** "in many DD estimates, the truly affected group is often only a fraction of the treatment group, meaning that a measured 2 percent effect on the full sample would indicate a much larger effect on the truly affected subsample." (pp.260-261)
- **Unbalanced panels, missing state-year cells, empty cells, weights, failed regressions:** Not discussed in paper.

**Reference implementation(s):** None is distributed with the article. Software named in the paper: the "cluster" command in STATA for both the state-year and the state-level arbitrary variance-covariance matrix (footnotes 13 and 24, pp.258, 271), and "the 'xtgls' command" in stata for the AR(k) corrections (Table IV note e, p.263). For the block bootstrap: "Implementing block bootstrap does require a limited amount of programming. The codes generated for this study are available upon request." (footnote 21, p.267). No code for the placebo-law generator is referenced.

### Paper-derived requirements

*Only items the paper specifies for its own exercise or explicitly recommends. These constrain what may be attributed to this source; they do not constrain the API surface.*

- [ ] Placebo laws are assigned on real outcome data in which the laws are fictitious, so the true effect is zero by construction (pp.251, 256)
- [ ] A placebo law is defined by a randomly drawn passage year and a randomly drawn set of treated states; the intervention dummy equals 1 in treated states after the intervention date and 0 otherwise (pp.255-256)
- [ ] The passage year is drawn uniformly from an interior window that leaves observations before and after it (1985-1995 within 1979-1999 in the paper) (p.256; footnote 9)
- [ ] The number of treated states is fixed (exactly half in the paper) and their identity is random (p.256)
- [ ] Each replication estimates the DD regression with state and year fixed effects (equation (1), p.250, or its state-year aggregated analogue, footnote 14, p.258) and records the estimate and its standard error (p.256)
- [ ] Micro data are aggregated by residualizing the outcome on individual controls and averaging the residuals by state and year (footnote 14, p.258)
- [ ] For conventional OLS standard errors, the rejection rate is the fraction of replications in which the absolute t-statistic exceeds 1.96 - a two-sided test at the 5 percent level (p.256; table notes). For the other inference methods, a replication counts as a rejection under that method's own stated decision rule (block bootstrap: "if 95 percent of the t_r are smaller than t", p.265; aggregation: a t-statistic "adjusted to take into account the smaller number of observations", p.267)
- [ ] The number of replications is at least 200 and typically 400 per cell, and a standard error of each rejection rate is reported, "computed using the number of simulations" (table notes); the one exception is Table II row 8, which prints a no-effect rejection rate of 0 with no standard error
- [ ] With outcomes held fixed across replications the correct rejection rate "is not known"; the paper's preferred version resamples whole states with replacement before assigning the law, preserving each state's own time series (footnote 12, p.256; p.258)
- [ ] Power is assessed by replacing the outcome with the outcome plus `I_st` times a chosen effect (0.02 log points in the paper), uniform across treated states, redrawing laws each replication, and counting rejections of the null of no effect (pp.259, 261; footnote 16)
- [ ] Rejection rates are reported side by side under "No effect" and under the injected effect, for each inference method, and by number of states and number of periods (Tables II-VIII)
- [ ] The outcome's serial correlation is summarized by first-, second- and third-order autocorrelation coefficients of state-year residuals net of state and year dummies, estimated by pooled OLS on lagged residuals (p.255)
- [ ] A fully simulated counterpart (AR(1) with normal disturbances calibrated to the real data's variance decomposition, new data each replication) is run as a check that real-data results are not driven by actual laws or ambient trends (p.260)
- [ ] Recommended to practitioners: examine residuals and test for serial correlation; compute standard errors robust to serial correlation as standard practice (p.274)
- [ ] Remedies the paper endorses, with their stated domains: block bootstrap when the number of groups is large enough; collapsing to pre and post periods (with a small-sample adjustment of the t-statistic) even for small numbers of states, at low power; an arbitrary within-state variance-covariance matrix when the number of groups is sufficiently large (pp.273-274)

### What the paper does NOT provide

**The paper's simulation evidence is how often a fixed OLS DD specification rejects the null, under alternative ways of computing the standard error or the test.** Besides those rejection frequencies it tabulates only autocorrelation estimates and the Table I survey statistics. Each item below was checked against journal pp. 249-275:

- **(a) No estimator other than OLS DD with state and year fixed effects.** The exhibits estimate equation (1) or its aggregated analogue, by OLS except where noted in the qualifications below. The "alternative estimators" of Section IV are alternative standard-error or test constructions. Two qualifications, as the paper leaves them: the aggregation methods re-estimate the coefficient on a collapsed two-period panel (p.267), and the AR(k) corrections are run through a GLS command (Table IV note e) without the paper saying whether the reported coefficient is the OLS or the GLS one. IV/GMM techniques are explicitly not explored (footnote 19, p.265), and GLS or GMM estimation is named only as a hope for future work (p.274).
- **(b) No comparison or ranking of estimators.** What is compared is inference methods for one estimator. The paper states that it sets the point estimate aside: "We assume away biases in estimating the intervention's effect" (p.250).
- **(c) No bias or RMSE of the point estimate.** No mean, bias, variance, MSE or RMSE of `beta_hat` is tabulated under either the no-effect or the 2 percent design. The only point-estimate summary is a description of the placebo estimates that happened to be significant: "The average absolute value of the estimated 'significant effects' in the wage regressions is about .02 [...] Nearly 60 percent of the significant estimates fall in the 1 to 2 percent range. About 30 percent fall in the 2 to 3 percent range, and the remaining 10 percent are larger than 3 percent." (p.260). The claim that standard errors "understate the standard deviation of the estimators" (abstract, p.249) is supported through rejection rates; neither the average standard error nor the standard deviation of the estimates is printed.
- **(d) No confidence-interval coverage.** No coverage rate is reported. Confidence intervals are mentioned once in passing (p.250) and the bootstrap decision is described "at a 95 percent confidence level" (p.265). Under the 2 percent design the tabulated quantity is rejection of `beta = 0`, not whether an interval contains 0.02.
- **(e) No minimum detectable effect.** Power is reported at one alternative (2 percent) only. No effect-size grid, power curve, target power, or detectable-effect calculation appears.
- **(f) No held-out periods.** No split-sample, out-of-sample or holdout validation. The 1985-1995 passage-year window exists only "to ensure having enough observations prior and post-intervention" (footnote 9, p.256).
- **(g) No staggered adoption with heterogeneous effects.** Staggered placebo dates appear in Table VI and footnotes 10 and 22, always with a uniform injected effect. Heterogeneity is deferred in one sentence: "For some practical applications, one might also be interested in cases where the treatment effect is heterogeneous." (footnote 16, p.259)
- **(h) No recipe for a practitioner to run placebo laws on their own data.** The placebo-law exercise is the authors' own experiment on the CPS, described in prose. The practitioner-facing advice is to "more carefully examine residuals as well as perform simple tests of serial correlation" and to compute standard errors robust to serial correlation (p.274). Footnote 11 (p.256) notes that a placebo distribution "could be used to form a randomization inference test" but gives no procedure.
- **No size-adjusted power.** The "2% effect" column is each test's raw rejection frequency; no correction for a test's own size distortion is applied.
- **No evaluated randomization-inference or permutation test.** Footnote 11 is a remark only.
- **No variation in the share of treated states** in the published tables; half are always treated. The case of very few treated states is not analyzed (footnote 10 defers alternatives to the working-paper version).
- **No effect dynamics.** The injected effect is a constant level shift; no anticipation, ramp-up, decay or event-time profile is simulated, and no event-study specification is estimated.
- **No covariate-specification comparison.** Controls are fixed (four education dummies and a quartic in age).
- **No guidance on the number of replications or bootstrap draws** beyond the counts used.
- **No rule for the acceptable rejection rate.** Results are described qualitatively ("performs well", "quite well", "a little worse"); no tolerance around 5 percent is defined.
- **No printed formula** for the parametric AR(1) standard error, the empirical variance-covariance estimator, the small-sample t adjustment under aggregation, the standard error of a rejection rate, or the simulated data generating process.
- **No treatment of unbalanced panels, missing cells, or sampling weights.**
- **No theorem or proof.** Consistency statements are attributed to cited work (pp.269, 271).

**Any of these in a library implementation would be a library design choice requiring its own source. It would NOT be derived from this paper and must not be attributed to it.**

---

<!-- ============================================================
     END of registry-candidate block. Everything BELOW this
     marker is audit material (transcribed tables, quotes,
     implementation notes, gaps) and is NOT part of the proposed
     REGISTRY.md text.
     ============================================================ -->

## Results

Exhibit map (what each table holds fixed and what it varies):

| Exhibit | Page | Data | Varies | Inference method |
|---|---|---|---|---|
| Table I | p.253 | 92 published DD papers | - | Tabulation of practice, no simulation |
| Table II | p.257 | CPS micro and aggregate; AR(1) | Aggregation, clustering, resampling, law type, outcome, `rho` | Conventional OLS (row 2: clustered at state-year) |
| Table III | p.262 | CPS aggregate; AR(1) `rho` = .8 | `N` (50, 20, 10, 6) and `T` (21, 11, 7, 5, 3, 2; 50 for AR(1)) | Conventional OLS |
| Table IV | p.263 | CPS aggregate; AR(1), AR(2), AR(1) + white noise | Data generating process | OLS vs parametric AR(1) corrections |
| Table V | p.266 | CPS aggregate; AR(1) `rho` = .8 | `N` | OLS vs block bootstrap |
| Table VI | p.268 | CPS aggregate; AR(1) `rho` = .8 | `N`; common vs staggered laws | OLS vs simple aggregation vs residual aggregation |
| Table VII | p.270 | CPS aggregate; AR(1) `rho` = .8 | `N` | OLS vs empirical variance-covariance matrix |
| Table VIII | p.272 | CPS aggregate; AR(1) `rho` = .8 and `rho` = 0 | `N` | OLS vs arbitrary variance-covariance matrix ("Cluster") |

There are no figures in the paper.

### Table I (p.253): "Survey of DD Papers"

| Item | Value |
|---|---|
| Number of DD papers | 92 |
| Number with more than 2 periods of data | 69 |
| Number which collapse data into before-after | 4 |
| Number with potential serial correlation problem | 65 |
| Number with some serial correlation correction | 5 |
| - GLS | 4 |
| - Arbitrary variance-covariance matrix | 1 |
| **Distribution of time span for papers with more than 2 periods** | |
| Average | 16.5 |
| Percentile 1% | 3 |
| Percentile 5% | 3 |
| Percentile 10% | 4 |
| Percentile 25% | 5.75 |
| Percentile 50% | 11 |
| Percentile 75% | 21.5 |
| Percentile 90% | 36 |
| Percentile 95% | 51 |
| Percentile 99% | 83 |
| **Most commonly used dependent variables** | **Number** |
| Employment | 18 |
| Wages | 13 |
| Health/medical expenditure | 8 |
| Unemployment | 6 |
| Fertility/teen motherhood | 4 |
| Insurance | 4 |
| Poverty | 3 |
| Consumption/savings | 3 |
| **Informal techniques used to assess endogeneity** | **Number** |
| Graph dynamics of effect | 15 |
| See if effect is persistent | 2 |
| DDD | 11 |
| Include time trend specific to treated states | 7 |
| Look for effect prior to intervention | 3 |
| Include lagged dependent variable | 3 |
| Number with potential clustering problem | 80 |
| Number which deal with it | 36 |

Table note, verbatim: "Data come from a survey of all articles in six journals between 1990 and 2000: the American Economic Review, the Industrial Labor Relations Review, the Journal of Labor Economics, the Journal of Political Economy, the Journal of Public Economics, and the Quarterly Journal of Economics. We define an article as 'Difference-in-Difference' if it (1) examines the effect of a specific intervention and (2) uses units unaffected by the intervention as a control group."

**The survey as described in the text (Section II, pp.252-254):**
- Purpose: whether serial correlation has mattered "depends on (1) the typical length of the time series used, (2) the serial correlation of the most commonly used dependent variables, and (3) whether any procedures have been used to correct for it" (p.252).
- Inclusion rule: "we collected data on all DD papers published in six journals between 1990 and 2000. We classified a paper as 'DD' if it focuses on specific interventions and uses units unaffected by the law as a control group. We found 92 such papers." (p.252). Footnote 6: "we do not classify a paper that regresses wages on unemployment as a DD paper".
- What was tabulated: "the number of time periods, the nature of the dependent variable, and the technique(s) used to compute standard errors" (p.252), plus informal endogeneity checks and the grouped-error problem (p.254).
- Time span: "Sixty-nine of the 92 DD papers used more than two periods of data. Four of these papers began with more than two periods but collapsed the data into two effective periods: before and after. For the remaining 65 papers, the average number of periods used is 16.5, and the median is 11. More than 75 percent of the papers use more than five periods of data." (pp.252-253). Footnote 7: the 51 and 83 values "arise because several papers used monthly or quarterly data. When a paper used several data sets with different time spans, we only recorded the shortest span."
- Dependent variables: "The most commonly used variables are employment and wages. [...] Most of these variables are clearly highly autocorrelated." (p.254). "Interestingly, first-differenced variables, which likely exhibit negative autocorrelation, are quite uncommon in DD papers." (p.254)
- Corrections in use: "A vast majority of the surveyed papers do not address serial correlation at all. Only five papers explicitly deal with it. Of these, four use a parametric AR(k) correction. [...] The fifth allows for an arbitrary variance-covariance matrix within each state" (p.254).
- Grouped errors: "80 of the original 92 DD papers have a potential problem with grouped error terms as the unit of observation is more detailed than the level of variation [...]. Only 36 of these papers address this problem, either by clustering standard errors or by aggregating the data." (p.254)
- Informal endogeneity checks: "three papers include a lagged dependent variable in equation (1), seven include a time trend specific to the treated states, fifteen plot some graphs to examine the dynamics of the treatment effect, three examine whether there is an 'effect' before the law, two test whether the effect is persistent, and eleven formally attempt to do triple-differences (DDD) by finding another control group. In Bertrand, Duflo, and Mullainathan [2002] we show that most of these techniques do not alleviate the serial correlation issues." (p.254)

### Table II (p.257): "DD Rejection Rates for Placebo Laws"

**A. CPS DATA**

| Data | `rho_hat_1, rho_hat_2, rho_hat_3` | Modifications | Rejection rate: No effect | Rejection rate: 2% effect |
|---|---|---|---|---|
| 1) CPS micro, log wage | | | .675 (.027) | .855 (.020) |
| 2) CPS micro, log wage | | Cluster at state-year level | .44 (.029) | .74 (.025) |
| 3) CPS agg, log wage | .509, .440, .332 | | .435 (.029) | .72 (.026) |
| 4) CPS agg, log wage | .509, .440, .332 | Sampling w/replacement | .49 (.025) | .663 (.024) |
| 5) CPS agg, log wage | .509, .440, .332 | Serially uncorrelated laws | .05 (.011) | .988 (.006) |
| 6) CPS agg, employment | .470, .418, .367 | | .46 (.025) | .88 (.016) |
| 7) CPS agg, hours worked | .151, .114, .063 | | .265 (.022) | .280 (.022) |
| 8) CPS agg, changes in log wage | -.046, .032, .002 | | 0 (no standard error printed) | .978 (.007) |

**B. MONTE CARLO SIMULATIONS WITH SAMPLING FROM AR(1) DISTRIBUTION**

| Data | `rho` | Modifications | Rejection rate: No effect | Rejection rate: 2% effect |
|---|---|---|---|---|
| 9) AR(1) | .8 | | .373 (.028) | .725 (.026) |
| 10) AR(1) | 0 | | .053 (.013) | .783 (.024) |
| 11) AR(1) | .2 | | .123 (.019) | .738 (.025) |
| 12) AR(1) | .4 | | .19 (.023) | .713 (.026) |
| 13) AR(1) | .6 | | .333 (.027) | .700 (.026) |
| 14) AR(1) | -.4 | | .008 (.005) | .7 (.026) |

Table notes, verbatim:
- "a. Unless mentioned otherwise under 'Modifications,' reported in the last two columns are the OLS rejection rates of the null hypothesis of no effect (at the 5 percent significance level) on the intervention variable for randomly generated placebo interventions as described in text. The data used in the last column were altered to simulate a true 2 percent effect of the intervention. The number of simulations for each cell is at least 200 and typically 400."
- "b. CPS data are data for women between 25 and 50 in the fourth interview month of the Merged Outgoing Rotation Group for the years 1979 to 1999. In rows 3 to 8 of Panel A, data are aggregated to state-year level cells after controlling for demographic variables (four education dummies and a quartic in age). For each simulation in rows 1 through 3, we use the observed CPS data. For each simulation in rows 4 through 8, the data generating process is the state-level empirical distribution of the CPS data that puts a probability of 1/50 on the different states' outcomes (see text for details). For each simulation in Panel B, the data generating process is an AR(1) model with normal disturbances chosen to match the CPS state female wage variances (see text for details). rho_hat_i refer to the estimated autocorrelation parameter of lag i. rho refers to the autocorrelation parameter in the AR(1) model."
- "c. All regressions include, in addition to the intervention variable, state and year fixed effects. The individual level regressions also include demographic controls."

**Findings stated in the text (pp.256-260):**
- Row 1: "We find that the null of no effect is rejected a stunning 67.5 percent of the time." (p.256)
- Rows 2-3: "We still find a very high (44 percent) rejection rate. [...] Here again, we reject the null of no effect in about 44 percent of the regressions. So correlated shocks within state-year cells explain only part of the overrejection we observe in row 1." (p.258)
- Row 4: "OLS standard errors lead to reject the null hypothesis of no effect at the 5 percent significance level in 49 percent of the cases." (p.258). Under the 2 percent effect, "OLS rejects the null of no effect in 66 percent of the simulations (row 4, last column)." (p.259)
- Row 5: "In row 5 we see that the null of no effect is now rejected in only 5 percent of the cases." (p.259)
- Rows 6-8: "Here we repeat the Monte Carlo study (as in row 4) for three different variables in the CPS: employment, hours, and change in log wages. [...] the overrejection problem diminishes with the serial correlation in the dependent variable. As expected, when the estimate of the first-order autocorrelation is negative (row 8), we find that OLS lead us to reject the null of no effect in less than 5 percent of the simulations." (pp.259-260)
- Rows 9-14: "In row 9 we assume that the autocorrelation parameter of the AR(1) model (rho) equals .8. We find a rejection rate of 37 percent. In rows 10 through 14 we show that as rho goes down, the rejection rates fall. When rho is negative (row 14), there is underrejection." (p.260)
- Summary: "The results in Table II demonstrate that, in the presence of positive serial correlation, conventional DD estimation leads to gross overestimation of t-statistics and significance levels." (p.260)
- Magnitudes of the false positives: "the magnitudes of the estimates obtained in these false rejections do not seem out of line with what is regarded in the literature as 'significant' economic impacts. The average absolute value of the estimated 'significant effects' in the wage regressions is about .02, which corresponds roughly to a 2 percent effect. Nearly 60 percent of the significant estimates fall in the 1 to 2 percent range. About 30 percent fall in the 2 to 3 percent range, and the remaining 10 percent are larger than 3 percent. These magnitudes are large, considering that DD estimates are often presented as elasticities." (p.260)
- Headline figure as stated up front: "we find an 'effect' significant at the 5 percent level for up to 45 percent of the placebo interventions" (abstract, p.249); "in as much as 45 percent of the simulations" (p.251).

### Table III (p.262): "Varying N and T"

**A. CPS DATA**

| Data | N | T | Rejection rate: No effect | Rejection rate: 2% effect |
|---|---|---|---|---|
| 1) CPS aggregate | 50 | 21 | .49 (.025) | .663 (.024) |
| 2) CPS aggregate | 20 | 21 | .39 (.024) | .54 (.025) |
| 3) CPS aggregate | 10 | 21 | .443 (.025) | .510 (.025) |
| 4) CPS aggregate | 6 | 21 | .383 (.025) | .433 (.025) |
| 5) CPS aggregate | 50 | 11 | .20 (.020) | .638 (.024) |
| 6) CPS aggregate | 50 | 7 | .15 (.017) | .635 (.024) |
| 7) CPS aggregate | 50 | 5 | .078 (.013) | .5 (.025) |
| 8) CPS aggregate | 50 | 3 | .048 (.011) | .363 (.024) |
| 9) CPS aggregate | 50 | 2 | .055 (.011) | .28 (.022) |

**B. MONTE CARLO SIMULATIONS WITH SAMPLING FROM AR(1) DISTRIBUTION**

| Data | N | T | Rejection rate: No effect | Rejection rate: 2% effect |
|---|---|---|---|---|
| 10) AR(1), rho = .8 | 50 | 21 | .35 (.028) | .638 (.028) |
| 11) AR(1), rho = .8 | 20 | 21 | .35 (.028) | .538 (.029) |
| 12) AR(1), rho = .8 | 10 | 21 | .3975 (.028) | .505 (.029) |
| 13) AR(1), rho = .8 | 6 | 21 | .393 (.028) | .5 (.029) |
| 14) AR(1), rho = .8 | 50 | 11 | .335 (.027) | .588 (.028) |
| 15) AR(1), rho = .8 | 50 | 5 | .175 (.022) | .5525 (.029) |
| 16) AR(1), rho = .8 | 50 | 3 | .09 (.017) | .435 (.029) |
| 17) AR(1), rho = .8 | 50 | 50 | .4975 (.029) | .855 (.020) |

Table notes, verbatim:
- "a. Reported in the last two columns are the OLS rejection rates of the null hypothesis of no effect (at the 5 percent significance level) on the intervention variable for randomly generated placebo interventions as described in text. The data used in the last column were altered to simulate a true 2 percent effect of the intervention. The number of simulations for each cell is typically 400 and at least 200."
- "b. CPS data are data for women between 25 and 50 in the fourth interview month of the Merged Outgoing Rotation Group for the years 1979 to 1999. The dependent variable is log weekly earnings. Data are aggregated to state-year level cells after controlling for the demographic variables (four education dummies and a quartic in age). For each simulation in Panel A, the data generating process is the state-level empirical distribution of the CPS data that puts a probability of 1/50 on the different states' outcomes (see text for details). For each simulation in Panel B, the data generating process is an AR(1) model with normal disturbances chosen to match the CPS state female wage variances (see text for details). rho refers to the autocorrelation parameter in the AR(1) data generating process."
- "c. All regressions also include, in addition to the intervention variable, state and year fixed effects."
- "d. Standard errors are in parentheses and are computed using the number of simulations."
- "e. N refers to the number of states used in the simulation and T refers to the number of years."

**Findings stated in the text (p.261):**
- Motivation: "The stylized exercise above focused on data with 50 states and 21 time periods. Many DD papers use fewer states (or treated and control units), either because of data limitations or because of a desire to focus only on comparable controls. For similar reasons, several DD papers use fewer time periods."
- "Rows 1-4 and 10-13 show that varying the number of states does not change the extent of the overrejection. Rows 5-9 and 14-17 vary the number of years. As expected, overrejection falls as the time span gets shorter, but it does so at a rather slow rate. For example, even with only seven years of data, the rejection rate is 15 percent in the CPS-based simulations. Conditional on using more than two periods, around 60 percent of the DD papers in our survey use at least two periods. With five years of data the rejection rate varies between 8 percent (CPS) and 17 percent (AR(1), rho = 8). When T = 50, the rejection rate rises to nearly 50 percent in the simulations using an AR(1) model with rho = .8." (p.261; "at least two periods" and "rho = 8" are as printed)

### Table IV (p.263): "Parametric Solutions"

**A. CPS DATA**

| Data | Technique | Estimated `rho_hat_1` | Rejection rate: No effect | Rejection rate: 2% Effect |
|---|---|---|---|---|
| 1) CPS aggregate | OLS | | .49 (.025) | .663 (.024) |
| 2) CPS aggregate | Standard AR(1) correction | .381 | .24 (.021) | .66 (.024) |
| 3) CPS aggregate | AR(1) correction imposing rho = .8 | | .18 (.019) | .363 (.024) |

**B. OTHER DATA GENERATING PROCESSES**

| Data | Technique | Estimated `rho_hat_1` | Rejection rate: No effect | Rejection rate: 2% Effect |
|---|---|---|---|---|
| 4) AR(1), rho = .8 | OLS | | .373 (.028) | .765 (.024) |
| 5) AR(1), rho = .8 | Standard AR(1) correction | .622 | .205 (.023) | .715 (.026) |
| 6) AR(1), rho = .8 | AR(1) correction imposing rho = .8 | | .06 (.023) | .323 (.027) |
| 7) AR(2), rho_1 = .55, rho_2 = .35 | Standard AR(1) correction | .444 | .305 (.027) | .625 (.028) |
| 8) AR(1) + white noise, rho = .95, noise/signal = .13 | Standard AR(1) correction | .301 | .385 (.028) | .4 (.028) |

Table notes, verbatim:
- "a. Reported in the last two columns are the rejection rates of the null hypothesis of no effect (at the 5 percent significance level) on the intervention variable for randomly generated placebo interventions as described in text. The data used in the last column were altered to simulate a true 2 percent effect of the intervention. The number of simulations for each cell is typically 400 and at least 200."
- "b. CPS data are data for women between 25 and 50 in the fourth interview month of the Merged Outgoing Rotation Group for the years 1979 to 1999. The dependent variable is log weekly earnings. Data are aggregated to state-year level cells, after controlling for the demographic variables (four education dummies and a quartic in age). For each simulation in Panel A, the data generating process is the state-level empirical distribution of the CPS data that puts a probability of 1/50 on the different states' outcomes (see text for details). For each simulation in Panel B, the distributions from which the data are drawn are chosen to match the CPS state female wage variances (see text for details). 'AR(1) + white noise' is the sum of an AR(1) plus an i.i.d. process, where the autocorrelation for the AR(1) component is given by rho and the relative variance of the components is given by the noise to signal ratio."
- "c. All regressions include, in addition to the intervention variable, state and year fixed effects."
- "d. Standard errors are in parentheses and are computed using the number of simulations."
- "e. The AR(k) corrections are implemented in stata using the 'xtgls' command."

**Findings stated in the text (pp.263-265):**
- Row 2: "This technique does little to solve the serial correlation problem: the rejection rate stays high at 24 percent." (p.264)
- Row 3: "if we impose a first-order autocorrelation of .8 in the CPS data (row 3), the rejection rate only goes down to 16 percent, a very partial improvement." (p.264). The table prints .18 for this cell.
- Row 7: "We then correct the standard error assuming that the error term follows an AR(1) process. The rejection rate rises significantly with this misspecification of the autocorrelation structure (30.5 percent)." (p.264)
- Row 8: "When trying to correct the autocorrelation in these data by fitting an AR(1), we reject the null of no effect in about 39 percent of the cases." (p.264)
- "The parametric corrections we have explored do not appear to provide an easy solution for the applied researcher." (pp.264-265). Footnote 20 (p.265): "when we use the two 'reasonable' processes described above in the CPS data or in a Monte Carlo study based on the empirical distribution of the CPS data, the rejection rates remained high."
- Survey link: "This is the method that was followed in four of the five surveyed DD papers that attempted to deal with serial correlation." (p.263)

### Table V (p.266): "Block Bootstrap"

**A. CPS DATA**

| Data | Technique | N | Rejection rate: No effect | Rejection rate: 2% effect |
|---|---|---|---|---|
| 1) CPS aggregate | OLS | 50 | .43 (.025) | .735 (.022) |
| 2) CPS aggregate | Block bootstrap | 50 | .065 (.013) | .26 (.022) |
| 3) CPS aggregate | OLS | 20 | .385 (.022) | .595 (.025) |
| 4) CPS aggregate | Block bootstrap | 20 | .13 (.017) | .19 (.020) |
| 5) CPS aggregate | OLS | 10 | .385 (.024) | .48 (.024) |
| 6) CPS aggregate | Block bootstrap | 10 | .225 (.021) | .25 (.022) |
| 7) CPS aggregate | OLS | 6 | .48 (.025) | .435 (.025) |
| 8) CPS aggregate | Block bootstrap | 6 | .435 (.022) | .375 (.025) |

**B. AR(1) DISTRIBUTION**

| Data | Technique | N | Rejection rate: No effect | Rejection rate: 2% effect |
|---|---|---|---|---|
| 9) AR(1), rho = .8 | OLS | 50 | .44 (.035) | .70 (.032) |
| 10) AR(1), rho = .8 | Block bootstrap | 50 | .05 (.015) | .25 (.031) |

Table notes, verbatim:
- "a. Reported in the last two columns are the rejection rates of the null hypothesis of no effect (at the 5 percent significance level) on the intervention variable for randomly generated placebo interventions as described in text. The data used in the last column were altered to simulate a true 2 percent effect of the intervention. The number of simulations for each cell is typically 400 and at least 200. The bootstraps involve 400 repetitions."
- "b. CPS data are data for women between 25 and 50 in the fourth interview month of the Merged Outgoing Rotation Group for the years 1979 to 1999. The dependent variable is log weekly earnings. Data are aggregated to state-year level cells after controlling for the demographic variables (four education dummies and a quartic in age). For each simulation we draw each state's vector from these data with replacement. See text for details. The AR(1) distribution is chosen to match the CPS state female wage variances (see text for details)."
- "c. All CPS regressions also include, in addition to the intervention variable, state and year fixed effects."
- "d. Standard errors are in parentheses and are computed using the number of simulations."

**Findings stated in the text (pp.265-267):**
- "This correction method presents a major improvement over the parametric techniques discussed before. When N equals 50, the rejection rate of the null of no effect is 6.5 percent in data drawn from the CPS and 5 percent in data drawn from an AR(1) model. When there is a 2 percent effect, the null of no effect is rejected in 26 percent of the cases in the CPS data and in 25 percent of the cases in the AR(1) data. However, the method performs less well when the number of states declines. The rejection rate is 13 percent with twenty states and 23 percent with ten states. The power of this test also declines quite fast. With twenty states, the null of no effect is rejected in only 19 percent of the cases when there is a 2 percent effect." (pp.265-266)
- "While block bootstrap provides a reliable solution to the serial correlation problem when the number of groups is large enough, this technique is rarely used in practice by applied researchers, perhaps because it is not immediate to implement." (p.267)

### Table VI (p.268): "Ignoring Time Series Data"

**A. CPS DATA**

| Data | Technique | N | Rejection rate: No effect | Rejection rate: 2% effect |
|---|---|---|---|---|
| 1) CPS agg | OLS | 50 | .49 (.025) | .663 (.024) |
| 2) CPS agg | Simple aggregation | 50 | .053 (.011) | .163 (.018) |
| 3) CPS agg | Residual aggregation | 50 | .058 (.011) | .173 (.019) |
| 4) CPS agg, staggered laws | Residual aggregation | 50 | .048 (.011) | .363 (.024) |
| 5) CPS agg | OLS | 20 | .39 (.025) | .54 (.025) |
| 6) CPS agg | Simple aggregation | 20 | .050 (.011) | .088 (.014) |
| 7) CPS agg | Residual aggregation | 20 | .06 (.011) | .183 (.019) |
| 8) CPS agg, staggered laws | Residual aggregation | 20 | .048 (.011) | .130 (.017) |
| 9) CPS agg | OLS | 10 | .443 (.025) | .51 (.025) |
| 10) CPS agg | Simple aggregation | 10 | .053 (.011) | .065 (.012) |
| 11) CPS agg | Residual aggregation | 10 | .093 (.014) | .178 (.019) |
| 12) CPS agg, staggered laws | Residual aggregation | 10 | .088 (.014) | .128 (.017) |
| 13) CPS agg | OLS | 6 | .383 (.024) | .433 (.024) |
| 14) CPS agg | Simple aggregation | 6 | .068 (.013) | .07 (.013) |
| 15) CPS agg | Residual aggregation | 6 | .11 (.016) | .123 (.016) |
| 16) CPS agg, staggered laws | Residual aggregation | 6 | .09 (.014) | .138 (.017) |

**B. AR(1) DISTRIBUTION**

| Data | Technique | N | Rejection rate: No effect | Rejection rate: 2% effect |
|---|---|---|---|---|
| 17) AR(1), rho = .8 | Simple aggregation | 50 | .050 (.013) | .243 (.025) |
| 18) AR(1), rho = .8 | Residual aggregation | 50 | .045 (.012) | .235 (.024) |
| 19) AR(1), rho = .8, staggered laws | Residual aggregation | 50 | .075 (.015) | .355 (.028) |

Table notes, verbatim:
- "a. Reported in the last two columns are the rejection rates of the null hypothesis of no effect (at the 5 percent significance level) on the intervention variable for randomly generated placebo interventions as described in text. The data used in the last column were altered to simulate a true 2 percent effect of the intervention. The number of simulations for each cell is typically 400 and at least 200."
- "b. CPS data are data for women between 25 and 50 in the fourth interview month of the Merged Outgoing Rotation Group for the years 1979 to 1999. The dependent variable is log weekly earnings. Data are aggregated to state-year level cells after controlling for demographic variables (four education dummies and a quartic in age). For each simulation we draw each state's vector from these data with replacement. See text for details. The AR(1) distribution is chosen to match the CPS state female wage variances (see text for details)."
- "c. All regressions also include, in addition to the intervention variable, state and year fixed effects."
- "d. Standard errors are in parentheses and are computed using the number of simulations."

**Findings stated in the text (pp.267-269):**
- "The rejection rate when N equals 50 is now 5.3 percent (row 2)." (p.267)
- "This procedure does as well as the simple aggregation (row 3 versus row 2) for laws that are all passed at the same time. It also does well when the laws are staggered over time (row 4)." (p.267)
- "Residual aggregation performs a little worse, but the overrejection remains relatively small. For example, for ten states, the rejection rate is 5.3 percent under the simple aggregation method (row 10) and about 9 percent under the residual aggregation method (row 11)." (p.267)
- "The downside of these procedures (both raw and residual aggregation) is that their power is quite low and diminishes fast with sample size. In the CPS simulations with a 2 percent effect, simple aggregation rejects the null only 16 percent of the time with 50 states (row 2), 8.8 percent of time with 20 states (row 6), and 6.5 percent of the time with 10 states (row 10)." (p.269)

### Table VII (p.270): "Empirical Variance-Covariance Matrix"

**A. CPS DATA**

| Data | Technique | N | Rejection rate: No effect | Rejection rate: 2% effect |
|---|---|---|---|---|
| 1) CPS aggregate | OLS | 50 | .49 (.025) | .663 (.024) |
| 2) CPS aggregate | Empirical variance | 50 | .055 (.011) | .243 (.021) |
| 3) CPS aggregate | OLS | 20 | .39 (.024) | .54 (.025) |
| 4) CPS aggregate | Empirical variance | 20 | .08 (.013) | .138 (.017) |
| 5) CPS aggregate | OLS | 10 | .443 (.025) | .510 (.025) |
| 6) CPS aggregate | Empirical variance | 10 | .105 (.015) | .145 (.018) |
| 7) CPS aggregate | OLS | 6 | .383 (.025) | .433 (.025) |
| 8) CPS aggregate | Empirical variance | 6 | .153 (.018) | .185 (.019) |

**B. AR(1) DISTRIBUTION**

| Data | Technique | N | Rejection rate: No effect | Rejection rate: 2% effect |
|---|---|---|---|---|
| 9) AR(1), rho = .8 | Empirical variance | 50 | .07 (.017) | .25 (.030) |

Table notes, verbatim:
- "a. Reported in the last two columns are the rejection rates of the null hypothesis of no effect (at the 5 percent significance level) on the intervention variable for randomly generated placebo interventions as described in text. The data used in the last column were altered to simulate a true 2 percent effect of the intervention. The number of simulations for each cell is typically 400 and at least 200."
- "b. CPS data are data for women between 25 and 50 in the fourth interview month of the Merged Outgoing Rotation Group for the years 1979 to 1999. The dependent variable is log weekly earnings. Data are aggregated to state-year level cells after controlling for demographic variables (four education dummies and a quartic age). For each simulation we draw each state's vector from these data with replacement. See text for details. The AR(1) distribution is chosen to match the CPS state female wage variances (see text for details)."
- "c. All regressions include, in addition to the intervention variable, state and year fixed effects."
- "d. Standard errors are in parentheses and are computed using the number of simulations."

**Findings stated in the text (p.269):**
- "The method performs well when the number of states is large (N = 50). The rejection rate we obtain in this case is 5.5 percent in the CPS (row 2) and 7 percent in the Monte Carlo simulations (row 9). Its power when N = 50 is comparable to the power of the block bootstrap method. In the Monte Carlo study based on the empirical distribution of the CPS, we reject the null of no effect in 24 percent of the simulations when there is a 2 percent effect."
- "However, as Table VII also indicates, this method performs more poorly for small sample sizes. As the number of states drops, the rejection rate of the null of no effect increases. For N = 10, this correction method leads us to reject the null of no effect in 8 percent of the cases; for N = 6, the rejection rate is 15 percent." The table prints .08 at N = 20 and .105 at N = 10.

### Table VIII (p.272): "Arbitrary Variance-Covariance Matrix"

**A. CPS DATA**

| Data | Technique | N | Rejection rate: No effect | Rejection rate: 2% effect |
|---|---|---|---|---|
| 1) CPS aggregate | OLS | 50 | .49 (.025) | .663 (.024) |
| 2) CPS aggregate | Cluster | 50 | .063 (.012) | .268 (.022) |
| 3) CPS aggregate | OLS | 20 | .385 (.024) | .535 (.025) |
| 4) CPS aggregate | Cluster | 20 | .058 (.011) | .13 (.017) |
| 5) CPS aggregate | OLS | 10 | .443 (.025) | .51 (.025) |
| 6) CPS aggregate | Cluster | 10 | .08 (.014) | .12 (.016) |
| 7) CPS aggregate | OLS | 6 | .383 (.024) | .433 (.025) |
| 8) CPS aggregate | Cluster | 6 | .115 (.016) | .118 (.016) |

**B. AR(1) DISTRIBUTION**

| Data | Technique | N | Rejection rate: No effect | Rejection rate: 2% effect |
|---|---|---|---|---|
| 9) AR(1), rho = .8 | Cluster | 50 | .045 (.012) | .275 (.026) |
| 10) AR(1), rho = 0 | Cluster | 50 | .035 (.011) | .74 (.025) |

Table notes, verbatim:
- "a. Reported in the last two columns are the rejection rates of the null hypothesis of no effect (at the 5 percent significance level) on the intervention variable for randomly generated placebo interventions as described in text. The data used in the last column were altered to simulate a true 2 percent effect of the intervention. The number of simulations for each cell is typically 400 and at least 200."
- "b. CPS data are data for women between 25 and 50 in the fourth interview month of the Merged Outgoing Rotation Group for the years 1979 to 1999. The dependent variable is log weekly earnings. Data are aggregated to state-year level cells after controlling for demographic variables (four education dummies and a quartic in age). For each simulation we draw each state's vector from these data with replacement. See text for details. The AR(1) distribution is chosen to match the CPS state female wage variances (see text for details)."
- "c. All regressions include, in addition to the intervention variable, state and year fixed effects."
- "d. Standard errors are in parentheses and are computed using the number of simulations."

**Findings stated in the text (pp.271-272):**
- "Despite its generality, the arbitrary variance-covariance matrix does quite well. The rejection rate in data drawn from CPS is 6.3 percent when N = 50 (row 2). With respect to power, we saw in Tables II and IV that with the correct covariance matrix, the rejection rate in the case of a 2 percent effect was 78 percent in a Monte Carlo simulation with no autocorrelation and 32 percent in AR(1) data with rho = .8. The arbitrary variance-covariance matrix comes near these upper bounds, achieving rejection rates of 74 percent (row 10) and 27.5 percent (row 9), respectively." (p.271)
- "Again, however, rejection rates increase significantly above 5 percent when the number of states declines: 11.5 percent with six states (row 8), 8 percent with ten states (row 6). The extent of the overrejection in small samples is comparable to that obtained for the empirical variance-covariance matrix correction method, less extreme than with block bootstrap, but higher than with the time series aggregation." (pp.271-272)

### Cross-table comparison at each N (reviewer compilation of printed cells, CPS-based rows)

Compiled from Tables V-VIII; "no effect / 2% effect". The paper prints no such combined table.

| Method | N = 50 | N = 20 | N = 10 | N = 6 |
|---|---|---|---|---|
| OLS (as printed in Table III / VI / VII) | .49 / .663 | .39 / .54 | .443 / .510 | .383 / .433 |
| OLS (as printed in Table V) | .43 / .735 | .385 / .595 | .385 / .48 | .48 / .435 |
| Block bootstrap | .065 / .26 | .13 / .19 | .225 / .25 | .435 / .375 |
| Simple aggregation | .053 / .163 | .050 / .088 | .053 / .065 | .068 / .07 |
| Residual aggregation | .058 / .173 | .06 / .183 | .093 / .178 | .11 / .123 |
| Residual aggregation, staggered laws | .048 / .363 | .048 / .130 | .088 / .128 | .09 / .138 |
| Empirical variance | .055 / .243 | .08 / .138 | .105 / .145 | .153 / .185 |
| Cluster (arbitrary variance) | .063 / .268 | .058 / .13 | .08 / .12 | .115 / .118 |

Reviewer note: at N = 6 the block bootstrap row and the Table V OLS row both print a "2% effect" rejection rate BELOW the "No effect" rate (.375 vs .435; .435 vs .48). The paper does not comment on this.

### Summary and conclusion as stated by the authors (Section IV.F, pp.272-273; Section V, pp.273-274)

> "First, 'naive' parametric corrections, which do not take into account the bias in the estimation of the autocorrelation parameters in short time series, perform poorly [Nickell 1981]. Furthermore, the time series lengths typical to DD applications are generally too short to reliably estimate more flexible data generating processes and misspecification of the process leads to inconsistent standard errors [Greene 2002]." (p.273)

> "Second, the arbitrary and empirical variance-covariance matrix corrections perform well in large samples, but not as well when the number of groups becomes small." (p.273). The paper relates this to cited results that the small-sample bias of White standard errors "is larger for variables that are constant or nearly constant within cluster (which is the case of the treatment variables in the DD model)" and that both estimators "perform well in finite sample when N equals 50, but are biased when N equals 10" (p.273).

> "Finally, aggregating the time series information performs well even for small number of states, which reflects the fact that the significance threshold can be adjusted for the small effective sample size [Donald and Lang 2001]. However, these aggregation techniques have relatively low power." (p.273)

> "Our study suggests that, because of serial correlation, conventional DD standard errors may grossly understate the standard deviation of the estimated treatment effects, leading to serious overestimation of t-statistics and significance levels. Since a large fraction of the published DD papers we surveyed report t-statistics around 2, our results suggest that some of these findings may not be as significant as previously thought if the outcome variables under study are serially correlated." (p.273)

> "We show that block bootstrap can be used to compute consistent standard errors when the number of groups is sufficiently large. Moreover, we show that a few techniques that are readily available in standard econometrics packages also provide viable solutions for the applied researcher. Collapsing the data into pre- and postperiods produce consistent standard errors, even when the number of states is small (though the power of this test declines fast). Allowing for an arbitrary autocorrelation process when computing the standard errors is also a viable solution when the number of groups is sufficiently large." (p.274)

Abstract summary of the remedies: "Econometric corrections that place a specific parametric form on the time-series process do not perform well. Bootstrap (taking into account the autocorrelation of the data) works well when the number of states is large enough. Two corrections based on asymptotic approximation of the variance-covariance matrix work well for moderate numbers of states and one correction that collapses the time series information into a 'pre'- and 'post'-period and explicitly takes into account the effective sample size works well even for small numbers of states." (p.249)

---

## Practical Recommendations

All quotes verbatim with journal page numbers.

1. **Check for serial correlation.**
   - "We hope that our study provides some motivation for the practitioners who estimate DD models to more carefully examine residuals as well as perform simple tests of serial correlation." (p.274)

2. **Make serial-correlation-robust standard errors standard practice.**
   - "Because computing standard errors that are robust to serial correlation appears relatively easy to implement in most cases, it should become standard practice in applied work." (p.274)

3. **With many groups: block bootstrap or an arbitrary within-state variance-covariance matrix.**
   - "block bootstrap can be used to compute consistent standard errors when the number of groups is sufficiently large" (p.274)
   - "Allowing for an arbitrary autocorrelation process when computing the standard errors is also a viable solution when the number of groups is sufficiently large." (p.274)
   - Clusters must be whole states: "choosing entire states (and not only state-year cells) as clusters" (footnote 24, p.271).

4. **With few groups: collapse to pre and post, accepting low power.**
   - "If one adjusts the t-statistics for the small number of observations in the regression, this correction works well even when the number of groups is relatively small (e.g., ten states)." (p.252)
   - "Collapsing the data into pre- and postperiods produce consistent standard errors, even when the number of states is small (though the power of this test declines fast)." (p.274)

5. **Do not rely on parametric AR(k) corrections.**
   - "this correction does very little in practice in the way of correcting standard errors" (p.254)
   - "The parametric corrections we have explored do not appear to provide an easy solution for the applied researcher." (pp.264-265)

6. **Clustering at the state-year level is not enough.**
   - "correlated shocks within state-year cells explain only part of the overrejection we observe in row 1" (p.258)

7. **Most of the surveyed informal endogeneity checks do not alleviate the serial correlation problem.**
   - "In Bertrand, Duflo, and Mullainathan [2002] we show that most of these techniques do not alleviate the serial correlation issues." (p.254). The supporting analysis is not in the published article.

The paper gives prose procedures for its own experiments and for each correction method, including an explicit bootstrap procedure and decision rule ("In practice, we bootstrap the t-statistic as follows", p.265). It gives no checklist, and it does not instruct the reader to run placebo laws on their own data.

---

## Limitations Stated by the Authors

- **The expected rejection rate under fixed outcomes is unknown.** "the exact rejection rate we should expect is not known" (footnote 12, p.256; p.258).
- **Real data may contain real laws or trends.** "one might be concerned that we are by chance detecting actual laws or other relatively discrete changes" (p.260).
- **The AR(1) is illustrative.** "not because we think that such a process matches the female wage data the best" (footnote 17, p.260).
- **The 2 percent alternative is arbitrary.** "The choice of 2 percent as the alternative is admittedly somewhat arbitrary, but our conclusions on the relative power of each estimator do not depend on this specific value." (p.261). No results at other values are shown.
- **The injected effect is homogeneous.** Footnote 16 (p.259).
- **The asymptotic remedies fail with few groups.** Block bootstrap: "the method performs less well when the number of states declines" (p.266). Empirical matrix: "performs more poorly for small sample sizes" (p.269). Arbitrary matrix: "rejection rates increase significantly above 5 percent when the number of states declines" (p.271).
- **The empirical matrix requires homoskedasticity across states.** "it is only consistent under the assumption of cross-sectional homoskedasticity, an assumption that is likely to be violated in practice for many data sets" (p.270).
- **Aggregation costs power.** "their power is quite low and diminishes fast with sample size" (p.269).
- **Simple aggregation needs a common law date.** "this solution will work only for laws that are passed at the same time for all the treated states" (p.267).
- **Block bootstrap is harder to implement.** "it is not immediate to implement" (p.267).
- **Estimated autocorrelation coefficients are biased in short panels.** Footnote 8 (p.255); p.264.
- **Techniques not explored.** "We do not explore in this paper IV/GMM estimation techniques." (footnote 19, p.265). "We also hope that our study will contribute in generating further work on alternative estimation methods for DD models (such as GLS estimation or GMM estimation of dynamic panel data models) that could be more efficient in the presence of serial correlation." (p.274)
- **Bias is assumed away.** "We assume away biases in estimating the intervention's effect" (p.250). The validity of DD itself is conditional: "It is appropriate when the interventions are as good as random, conditional on time and group fixed effects." (p.250)

---

## Implementation Notes (audit notes - NOT registry-candidate)

### Data Structure Requirements

What the paper's exercise uses (not a proposed library contract):
- A real-data panel of groups by periods with one outcome and no treatment of interest: 50 states by 21 years, built from repeated cross sections of individuals (pp.250, 255). The paper works either with the micro data (Table II rows 1-2) or with state-year means of residualized outcomes (all later CPS-based rows; the AR(1) panels use generated data).
- Individual-level controls (four education dummies and a quartic in age): included directly in the micro-data regressions (p.256; Table II note c: "The individual level regressions also include demographic controls"), and used in the initial residualizing regression for the aggregated exercises (footnote 14, p.258).
- Enough periods on both sides of every admissible placebo passage date; the paper restricts the date to the interior eleven-year window 1985-1995 of the 21 sample years 1979-1999 (footnote 9, p.256). Reviewer arithmetic: a 1985-1995 date within 1979-1999 leaves at least six earlier years and at least four later years; whether the passage year itself counts as treated is not stated (see Gaps).
- A balanced state-by-year layout is implied by "50 * 21 = 1050" cells (p.255); unbalanced panels are not discussed in paper.
- For the resampled-states variant, each sampled state contributes its entire time series (p.258), so a state drawn twice appears as two separate groups. How duplicate draws are labeled for the state fixed effects is not discussed in paper.
- Sampling weights, minimum cell sizes, and treatment of women with non-positive earnings beyond their exclusion from the log-wage sample: Not discussed in paper.

### Computational Considerations

- **Replications.** At least 200 and typically 400 placebo draws per table cell (table notes). The micro-data version runs each regression on roughly 540,000 observations (p.255); the aggregate version runs on 1,050 state-year cells.
- **Two columns per cell.** Each design is run under no effect and again with the 2 percent effect added. Whether the same placebo-law draws are reused for the two columns (paired) or drawn afresh is not stated.
- **Block bootstrap cost (reviewer arithmetic, not printed).** 400 placebo draws with 200 bootstrap samples each is 80,000 regressions per cell and per column; with the 400 repetitions given in Table V note a it is 160,000.
- **Implied replication counts (reviewer reconstruction).** Reading the printed standard errors through `sqrt(p(1-p)/R)`: .49 (.025) implies `R` near 400; .675 (.027) implies `R` near 300; .44 (.035) in Table V row 9 implies `R` near 200. Table IV row 6 prints .06 (.023), which would imply `R` near 107 - below the stated minimum of 200; with `R = 400` the formula gives about .012. The paper does not explain this cell.
- **Resampling randomness.** The resampled-states variant draws a new set of 50 states every replication, so two sources of randomness (which states, which law) are combined in one rejection rate.
- **Random seed, parallelization, runtime:** Not discussed in paper.

### Tuning Parameters

| Parameter | Type | Default (BDM) | Selection Method |
|-----------|------|---------------|------------------|
| Replications per placebo cell | int | typically 400, >=200 | More draws -> tighter Monte Carlo estimate of size |
| Fake-date window | range | uniform 1985-1995 | Bounded to keep pre/post observations (footnote 9) |
| Fraction of groups treated | float | 0.5 (25 of 50) | Author choice; alternatives tried (footnote 10) |
| Power-test effect size `delta` | float | 0.02 (2%) | Set to the alternative of interest (p.259) |
| Block-bootstrap replications | int | 200 / 400 | See Gaps (text/table discrepancy) |
| State resampling probability | float | 1/50 (whole state vectors, w/ replacement) | Preserves within-group autocorrelation (p. 258) |

### Relation to Existing diff-diff Estimators

- **`PlaceboTests` (`diff_diff/diagnostics.py`) is the registered surface sourced from this paper.** `run_placebo_test` dispatches four diagnostics by `test_type`: `"fake_timing"` (a placebo treatment date), `"fake_group"` (fake treated units drawn from the controls), `"permutation"` (the randomization-inference idea of footnote 11, p.256) and `"leave_one_out"` (single-unit sensitivity, a library addition). REGISTRY `## PlaceboTests` is the authority for their behavior and validation status. REGISTRY also records that the paper's inference corrections (parametric AR, block bootstrap, cluster / arbitrary variance-covariance matrix, time aggregation) are out of scope for that diagnostic surface.
- **Distinct from per-estimator placebo/LOO** already documented elsewhere in the registry:
  SyntheticControl's donor-pool `leave_one_out()` (ADH 2015 Section 4) and in-time placebo,
  SyntheticDiD's jackknife leave-one-out effects and in-time placebo, HAD pretests, and DCDH
  placebo are separate implementations on their own estimators; the
  BDM-grounded `PlaceboTests` is the generic 2x2 DD battery.
- **Estimator and inference options.** `DifferenceInDifferences` and `TwoWayFixedEffects` accept `cluster=` (cluster-robust standard errors) and `inference="wild_bootstrap"` (wild cluster restricted bootstrap). `DifferenceInDifferences` requires an explicit `cluster=` for the wild bootstrap; with its default covariance settings `TwoWayFixedEffects` clusters at the unit level automatically when `cluster=None` (under `inference="analytical"`, an explicitly chosen `vcov_type` of `classical`, `hc2` or `hc3` turns that automatic clustering off; the wild bootstrap keeps it). By concept, cluster-robust standard errors with the group as the cluster correspond to the paper's arbitrary variance-covariance matrix with "entire states" as clusters (Section IV.E; footnote 24, p.271). The paper shows that clustering at the state-year cell level instead is not sufficient (Table II row 2), so under the paper's results the level of clustering matters. The paper's headline warning - conventional DD standard errors overreject when the outcome and the treatment indicator are both serially correlated - is the motivation for unit-level clustering in panel DD. Whether the library's cluster-robust formula matches the paper's printed `W` (in particular any finite-sample multiplier, which the paper does not print) was not checked in this review.
- **Wild bootstrap is not in the paper.** The paper's bootstrap is a block bootstrap that resamples whole states and bootstraps the absolute t-statistic (Section IV.B). A wild bootstrap is not evaluated or mentioned, so no rejection rate in this paper can be attributed to `inference="wild_bootstrap"`.
- **Remedies not checked.** Whether the library has the paper's block bootstrap, pre/post collapsing (simple or residual aggregation), parametric AR(1) correction, or empirical variance-covariance remedies has NOT been checked in this review; neither presence nor absence is asserted.
- **Existing placebo diagnostics, by function.** `diff_diff/diagnostics.py` has placebo diagnostics for the basic DiD estimator - individual fake-timing and fake-group placebo fits, a permutation procedure that repeatedly reassigns treatment and returns a permutation distribution, and a leave-one-out procedure that refits after dropping each treated unit: `run_placebo_test`, `placebo_timing_test`, `placebo_group_test`, `permutation_test`, and `leave_one_out_test`, each returning `PlaceboTestResults`, plus `run_all_placebo_tests`, which returns a dictionary of those results (or structured error dictionaries) keyed by test. Existing argument spellings include `fake_treatment_period`, `fake_treated_units`, `n_permutations`, `alpha`, `seed`. By concept and name only: the paper's randomly drawn passage year is a fake treatment period, its randomly drawn half of the states is a set of fake treated units, and its 5 percent level is `alpha`. The paper's output differs in kind from these: it repeats the placebo assignment hundreds of times and reports the rejection FREQUENCY of a test, paired with and without an injected effect. Whether any existing function's behavior matches a step of the paper's procedure was not re-checked in the 2026-11-22 audit; REGISTRY `## PlaceboTests` is the authority. The paper never mentions a random seed.
- **Existing simulation tools.** `diff_diff/power.py` has `simulate_power`, `simulate_mde`, and `simulate_sample_size`, which by default draw from an internal per-estimator registry of SYNTHETIC data generators and also accept a custom `data_generator` / `data_generator_kwargs` that bypasses the registry. Reviewer note: run on the default generators, that corresponds in kind to the paper's second Monte Carlo study (simulated AR(1) data, p.260), not to its placebo laws on real CPS data or to its resampled-states variant.
- **Paper-quantity-to-field map.** `SimulationPowerResults` carries `power`, `power_se`, `power_ci`, `rejection_rate`, `mean_estimate`, `std_estimate`, `mean_se`, `coverage`, `bias`, `rmse`, `true_effect`, `effect_sizes`, `powers`, `alpha`, `n_simulations`. Matching below is by concept and name only; whether each field's computation matches the paper's definition was not checked in this review.

  | Paper quantity | Where defined | Existing library field |
  |---|---|---|
  | Rejection rate under no effect: fraction of placebo draws rejected (absolute t-statistic above 1.96 for conventional OLS; each other method's own decision rule) | p.256; "No effect" column, Tables II-VIII | `rejection_rate` |
  | Rejection rate under the injected effect ("power against a specific alternative") | pp.259, 261; "2% effect" column, Tables II-VIII | `power` |
  | Standard error of a rejection rate ("computed using the number of simulations") | Table notes d, Tables III-VIII | `power_se` (by name, for power); none among the fields listed above for the no-effect rate |
  | Size of the injected effect (0.02 log points) | pp.259, 261; footnote 18 | `true_effect` |
  | Set of injected effect sizes | The paper uses a single value; ".0x" wording, p.259 | `effect_sizes` / `powers` exist for a grid; the paper reports no grid |
  | Significance level (5 percent; threshold 1.96) | p.256; table notes | `alpha` |
  | Number of placebo draws / simulations (at least 200, typically 400) | p.256; table notes | `n_simulations` |
  | Average absolute value of the SIGNIFICANT placebo estimates (about .02) and their distribution by size band | p.260 | none (`mean_estimate` is not conditional on significance and is not an absolute value) |
  | Power upper bound using the true variance-covariance matrix (32.5 percent; 78 percent) | footnote 18, p.261; p.271 | none |
  | Estimated autocorrelation coefficients of the outcome residuals (`rho_hat_1`, `rho_hat_2`, `rho_hat_3`) | p.255; Table II | none |
  | Estimated `rho_hat_1` from the AR(1) correction | Table IV | none |
  | Number of groups `N` and periods `T` of the design | Table III note e | none among the fields listed above |
  | Number of bootstrap samples (200 in text; 400 in table note) | p.265; Table V note a | none among the fields listed above |

- **Existing fields with no counterpart reported by the paper.** `bias`, `rmse`, `coverage`, `mean_estimate`, `std_estimate`, `mean_se` and `power_ci` name quantities the paper does not tabulate. Reviewer note: the paper's central verbal claim - that conventional standard errors "understate the standard deviation of the estimators" (abstract, p.249) - is a statement about the two concepts that `mean_se` and `std_estimate` name, but the paper supports it through rejection rates and prints neither quantity.
- **Reviewer note on how this source relates to the companion reviews.** The companion review of Coglianese and Fang (2026) found that its source supplies a placebo point-estimate MSE on real data with a single treated unit and no inference; the companion review of Blair, Cooper, Coppock and Humphreys (2019) found that its source supplies the framework and vocabulary (bias, RMSE, power, coverage) on fully simulated data. This paper supplies, on real data, the placebo-assignment procedure with multiple treated units, a rejection-rate (size) metric, and an additive effect-injection step that turns the same procedure into a power calculation. It supplies nothing for bias, RMSE, coverage, or minimum detectable effect, and it evaluates one estimator only.
- **Naming constraint.** The library has a standing constraint to reuse existing method / argument / result-field names wherever a concept already exists, so the "none" rows above are open naming questions and the matched rows are not. The API itself is to be decided in the implementation plan, not here.

---

## Gaps and Uncertainties

**Placebo-law definition**
- **Whether the passage year itself is treated.** `I_st` equals 1 "after the intervention date" (p.256); the paper does not say whether the drawn year is the first treated year or the last untreated year.
- **Sampling scheme for the treated states.** "we select exactly half the states (25) at random" (p.256) - no statement of the mechanism beyond that. In the resampled-states variant, whether "half of the states" refers to half of the 50 draws (so a state drawn twice could be treated in one copy and untreated in another) is not stated.
- **Half of N for small N.** For N = 20, 10 and 6 the paper does not restate the number treated; "half" is carried over from the base design by implication (reviewer reading).
- **How N is reduced.** For the CPS-based panel, Table III note b keeps the sampling scheme: each simulation draws from the empirical distribution that puts "a probability of 1/50 on the different states' outcomes", i.e. states are sampled with replacement from all 50 (p.262; procedure on p.258), with `N` "the number of states used in the simulation" (Table III note e). How duplicate draws of the same state are labelled is not explained.
- **How T is reduced or extended.** Which years are kept for T = 11, 7, 5, 3, 2, and how the passage year is drawn when the 1985-1995 window no longer fits (including T = 50 in the simulated data), are not stated.
- **Paired vs independent draws** across the "No effect" and "2% effect" columns, and across methods within a table, are not stated.
- **"at least 200" vs "typically 400".** The exact number of replications per cell is not printed; see Computational Considerations for the implied counts.
- **Serially uncorrelated laws.** Whether the ten dates are common to all treated states or drawn per state is not explicit ("we randomly select ten dates between 1979 and 1999", p.259).

**Effect injection**
- **Non-log outcomes.** The injection is defined for log weekly earnings (add 0.02). Section IV states that power is evaluated in both Monte Carlo studies by adding `I_st * 0.02` to the data (p.261), so the simulated AR(1) data receive the same additive 0.02. What a "2 percent effect" means on the outcome scale for employment, hours worked, or changes in log wage (Table II rows 6-8) is not explained. Table II row 7 (hours worked) prints a 2 percent rejection rate of .280 against a no-effect rate of .265, and row 8 (changes in log wage) prints .978; the paper does not discuss either.
- **Micro vs aggregate injection.** Whether the effect is added to individual outcomes before the residualizing and averaging step or directly to the state-year means is not stated.
- **No size adjustment.** Power for an overrejecting test is not corrected; the paper does not discuss this.
- **Upper-bound figures.** Footnote 18 gives 32.5 percent and p.271 gives "32 percent" and "78 percent" for the rejection rate "with the correct covariance matrix". p.271 attributes them to "Tables II and IV"; the matching printed cells are Table IV row 6 (.323) and Table II row 10 (.783), which are, respectively, an AR(1) correction imposing the true `rho` and OLS on serially uncorrelated data (reviewer reading). The 32.5 figure does not appear in any table.

**Text vs table mismatches (all as printed; none reconciled by the paper)**
- **Table IV row 3.** Text: "the rejection rate only goes down to 16 percent" (p.264); table: .18.
- **Table VII.** Text: "For N = 10 [...] 8 percent of the cases; for N = 6, the rejection rate is 15 percent" (p.269); table: .08 at N = 20, .105 at N = 10, .153 at N = 6.
- **Block bootstrap draws.** Text: "a large number (200) of bootstrap samples" (p.265); Table V note a: "The bootstraps involve 400 repetitions."
- **OLS benchmark rows differ across tables.** The CPS aggregate OLS row at N = 50 is .49 / .663 in Tables II-IV and VI-VIII but .43 / .735 in Table V; at N = 6 it is .383 / .433 elsewhere but .48 / .435 in Table V; at N = 20 Table VIII prints .385 / .535 against .39 / .54 elsewhere; at N = 10 Table V prints .385 / .48 against .443 / .510 elsewhere. Printed standard errors for nominally identical rows also differ in the third decimal (for example the N = 6 row: (.025)/(.025) in Tables III and VII, (.024)/(.024) in Table VI, (.024)/(.025) in Table VIII).
- **AR(1) `rho` = .8 OLS at N = 50, T = 21** is .373 / .725 (Table II row 9), .35 / .638 (Table III row 10), .373 / .765 (Table IV row 4) and .44 / .70 (Table V row 9). Reviewer note: these look like separate simulation runs; the paper does not say so.
- **Abstract vs Table II.** "up to 45 percent" (abstract, p.249) and "as much as 45 percent" (p.251) against printed no-effect rates of .44 and .435 (rows 2-3), .49 (row 4) and .675 (row 1).
- **p.261 sentences.** "around 60 percent of the DD papers in our survey use at least two periods" (conditional on more than two periods, as printed) and "(AR(1), rho = 8)" are transcribed as printed; both look like misprints, and the intended values are not stated.
- **Table II row 8** prints a no-effect rejection rate of "0" with no standard error.
- **Table IV row 6** prints a standard error of (.023) for a rate of .06, larger than the other no-effect standard errors for small rates (reviewer note; see Computational Considerations).
- **AR(1) + white noise.** Text: "the variance of the white noise is 13 percent of the total variance of the residual" (p.264); table: "noise/signal = .13", with the note that "the relative variance of the components is given by the noise to signal ratio". The two descriptions are not identical.
- **Table VII note b** prints "a quartic age" where the other tables print "a quartic in age".

**Placebo diagnostics (`PlaceboTests`)**
- **Footnote 12 is load-bearing for a permutation test.** Because treatment is randomized over a fixed set of outcomes, the induced test statistic is not standard normal (footnote 12, p.256). Reviewer reading: a permutation p-value therefore has to be read from the empirical placebo distribution, not from a t or normal table. The paper states the first sentence only; it evaluates no permutation test (footnote 11 is a remark). REGISTRY `## PlaceboTests` records the count-based p-value the library uses.
- **Fake-timing against a chosen pre-period, leave-one-out, and the permutation p-value floor are library extensions**, not procedures in the paper (see the checklist in the `## PlaceboTests` block).

**Inference methods**
- **Parametric AR(1) correction.** No formula. The text describes forming a variance-covariance matrix to "compute standard errors" (p.263), while the implementation is "xtgls" (Table IV note e); whether the reported coefficient is the OLS or the GLS estimate is not stated.
- **Block bootstrap.** The decision rule is stated in words only. The bootstrap is described for "50 matrices"; the number drawn when N is 20, 10 or 6 is not restated. Whether the bootstrap `SE(beta_hat_r)` is the conventional OLS standard error is implied, not stated.
- **Small-sample adjustment under aggregation.** "the t-statistic needs to be adjusted to take into account the smaller number of observations" (p.267) - the reference distribution and degrees of freedom are not printed, and it is not stated from which N the adjustment is applied.
- **Residual aggregation.** The final regression "in this two-period panel" (p.267) is not written out; with treated states only and two periods, the regressors and the degrees of freedom are not stated. Whether the first-stage regression includes the intervention dummy is not stated (the text lists "state fixed effects, year dummies, and any relevant covariates").
- **Simple aggregation.** For a common law date, how the averaged two-period regression treats the passage year is not stated.
- **Empirical variance-covariance matrix.** No estimator formula; the block element is described as "the correlation between epsilon_i and epsilon_{i-j}" with element index "(i, i + j)" as printed; whether variances are also estimated per period is not stated.
- **Arbitrary variance-covariance matrix.** The displayed formula has no finite-sample multiplier; the software command named in footnote 24 may apply one, and the paper does not say. The definitions under the formula read "state i" and "dependent variables" as printed.
- **Critical value for the non-bootstrap remedies.** The 1.96 threshold is stated for conventional OLS (p.256); the table notes say only "at the 5 percent significance level" for the other methods.
- **Methods not crossed with designs.** Staggered laws are tabulated only for residual aggregation; varying T only for OLS; the parametric corrections only in the base design (Table IV prints no N column).

**Simulated data**
- **No printed data generating equation.** The AR(1) design is described in words; the variance shares of state effects, year effects and the AR(1) component, and the innovation variance, are not printed.
- **Initial conditions / burn-in** of the simulated series: not stated.
- **How placebo laws are drawn in simulated data** (window for the passage year) is not restated.

**Survey**
- The list of the 92 papers is not given. The classification was made by the authors; no inter-coder procedure is described.
- "Number with potential serial correlation problem" (65) is defined operationally only as papers with more than two periods that did not collapse the data (reviewer reading of pp.252-253 and Table I: 69 - 4 = 65).
- Table I lists "GLS" (4) where the text says "a parametric AR(k) correction" (p.254).

**Material the article points to but does not contain**
- "See Bertrand, Duflo, and Mullainathan [2002] for details" on alternative placebo interventions, including changing the number of affected states and staggered laws (footnote 10, p.256), and on the informal endogeneity techniques (p.254).
- "unpublished appendix available from the authors" for fixed-treatment-year simulations (footnote 15, p.258).
- Block bootstrap code "available upon request" (footnote 21, p.267).
- None of these was reviewed; nothing in this file is drawn from them.

**Not developed by the paper**

*Each item below means the paper gives no procedure, formula, or evidence for it. It does not mean the words never occur: passing mentions are noted where found.*
- Any treatment-effect estimator other than OLS on equation (1) and its aggregated or collapsed versions.
- Bias, RMSE, coverage, minimum detectable effect, power curves, size-adjusted power.
- Heterogeneous or dynamic treatment effects; event-study specifications; anticipation.
- Unequal treated shares, a single treated unit, or very few treated units (deferred to the working-paper version by footnote 10).
- Cross-sectional (between-state) correlation of errors; the remedies address correlation within a state over time.
- Unbalanced panels, missing cells, sampling weights.
- Simulation evidence for units other than US states, frequencies other than annual, or outcomes other than the four CPS variables in Table II. (The methodological discussion is stated to generalize - "This discussion of course generalizes to other types of DD estimates", footnote 2, p.250 - and the Table I survey covers other outcomes and papers using "monthly or quarterly data", footnote 7, pp.253-254.)
- A procedure for a practitioner to run placebo laws on their own data, and any screen for real policy changes in the data used for placebo assignment.

**Bibliography**
- The reference list (pp.274-275) was not transcribed; citations in this review are given in the bracketed short form the paper uses in its text.
