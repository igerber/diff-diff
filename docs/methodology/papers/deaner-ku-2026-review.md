# Paper Review: Causal Duration Analysis with Diff-in-Diff

**Authors:** Ben Deaner, Hyejin Ku (University College London)
**Citation:** Deaner, B., & Ku, H. (2026). Causal Duration Analysis with Diff-in-Diff. arXiv preprint arXiv:2405.05220v2 (revise-and-resubmit, *Quantitative Economics*). https://arxiv.org/abs/2405.05220
**PDF reviewed:** https://arxiv.org/pdf/2405.05220v2 (arXiv v2, submitted 25 May 2026; title page dated May 26, 2026; 50 pages including Appendices A-E; SHA-256 `d40df031669d5ceb2a3fa39c9e904c708fd7f60e7f60f460ab4adab02945f639`; cached at `.workflow/sources/deaner-ku-2024-arxiv-2405.05220v2.pdf`)
**Reference code reviewed:** https://github.com/ben-deaner-teaching/Duration-DiD at commit `202e92ef222bf7d250c26c98e8fb42e151223c71` (main, 2026-05-11; six files, per-file digests in the inventory below; **no LICENSE file**)
**Status and version sources:** https://sites.google.com/site/hyejku/cv and https://bendeaner.wordpress.com/research/ (both accessed 2026-09-05; both list the paper as revise-and-resubmit at *Quantitative Economics*); https://arxiv.org/abs/2405.05220v1 (accessed 2026-09-05; v1 posted 8 May 2024, "46 pages, 7 figures, 2 tables")
**Review date:** 2026-09-05

> **Preprint status.** This is an unpublished arXiv preprint (v2, 25 May 2026; title page
> "May 26, 2026"; the acknowledgements thank anonymous referees). The revise-and-resubmit
> status at *Quantitative Economics* rests on the two authors' web pages cached on
> 2026-09-05 (secondary sources; re-check before citing in user-facing docs). No journal
> version exists. The library's usual rule is to source methodology from published papers;
> this review is an explicit exception, made for the method's direct relevance to
> absorbing-state outcomes that no shipped estimator handles. **All equation, assumption,
> theorem, remark, algorithm, table, figure and page references below are pinned to arXiv
> v2 ("p. N").** v1 (8 May 2024) has 46 pages, so v1 page numbers are NOT interchangeable
> with v2. The paper calls Appendices A-E the "supplementary appendix", but they are inside
> the same PDF (pp. 32-50). The bibliography begins mid-page 28 (eight entries on p. 28,
> Aalen 1989 through Ashenfelter & Card 2002) and ends at the top of p. 32. File-name note:
> the slug year (2026) follows the year printed on page 1 of the reviewed version; the arXiv
> identifier 2405.05220 dates from the 2024 first posting (repository precedent is mixed:
> `ciaccio-2024-review.md` reviews a 2025 v2 under a 2024 slug, `dechaisemartin-2026-review.md`
> reviews a v6 under a 2026 slug). Re-check every number against any later arXiv version or
> the published version before citing.

> **Source inventory.** All originals are cached under `.workflow/sources/` (relative
> paths; the coordinator includes them in review snapshots) and listed with the same
> digests in `.workflow/sources.json`.
>
> | File (relative to `.workflow/sources/`) | URL | Version | SHA-256 | Role |
> |---|---|---|---|---|
> | `deaner-ku-2024-arxiv-2405.05220v2.pdf` | https://arxiv.org/pdf/2405.05220v2 | arXiv v2, 25 May 2026, 50 pp. | `d40df031669d5ceb2a3fa39c9e904c708fd7f60e7f60f460ab4adab02945f639` | the paper |
> | `duration-did-repo-202e92ef/README.md` | raw.githubusercontent.com/ben-deaner-teaching/Duration-DiD/202e92ef.../README.md | commit 202e92ef | `d7bcfb98c9498a6701465d620cc2e3c6d1c65e0ce3377245d856228f147b9b7b` | usage documentation |
> | `duration-did-repo-202e92ef/durationdid.ado` | .../durationdid.ado | commit 202e92ef | `6e66e9f19ae3cf384d08209d52abcbaedcc85196f97e524e168ee7eaca58aa0e` | Stata/Mata command `durationdid v1.0` |
> | `duration-did-repo-202e92ef/durationDiD.m` | .../durationDiD.m | commit 202e92ef | `3449d12a4b9e2aca7f943493c9f303130f90d348d089fd88604ac352280f43df` | MATLAB estimator |
> | `duration-did-repo-202e92ef/UIP_application.do` | .../UIP_application.do | commit 202e92ef | `5a4ee15a3a8cbaabed725ea4973cd64f2c533581db85abb1cde45c907d09be2f` | Stata do-file, Section 4 application |
> | `duration-did-repo-202e92ef/UIP_application_revision_R2.m` | .../UIP_application_revision_R2.m | commit 202e92ef | `092ed6ab096a2127693221708df17f4e56ac52d5f235a82c1d2fffdd0fefa8d5` | MATLAB script, Section 4 application |
> | `duration-did-repo-202e92ef/montecarlo_revision.m` | .../montecarlo_revision.m | commit 202e92ef | `f7a01d3a46762c6d05eac88014c20dc60da468603367746dcb02da9fa1b5c022` | MATLAB script, Appendix C simulation |
> | `status-pages/hyejin-ku-cv-2026-09-05.html` | https://sites.google.com/site/hyejku/cv | as served 2026-09-05 20:52 UTC | `952a3e3a63eddcdaddcadd1cbc47e570e6ec9b50c0843ce0ea7ef589c52757fa` | R&R status (secondary) |
> | `status-pages/ben-deaner-research-2026-09-05.html` | https://bendeaner.wordpress.com/research/ | as served 2026-09-05 20:52 UTC | `e4f00e517a58e965ab1bdf1fe8ba02b39d65524a3c61851a22d240cede5fc1ef` | R&R status (secondary) |
> | `status-pages/arxiv-2405.05220v1-abs-2026-09-05.html` | https://arxiv.org/abs/2405.05220v1 | as served 2026-09-05 21:33 UTC | `cedd430d8f04242f22bbf89ae0594ae1914bacd3797b6f56a41fd3a7ae738812` | v1 date and page count |
>
> **Licence.** The repository has no LICENSE file, so all rights are reserved by default.
> The library must NOT port or translate the Stata or MATLAB source; the code may be used
> only as a black-box / equation-level reference (read to understand conventions, run
> independently to compare outputs if the maintainer has a licence for the tools). Both
> implementation files are two concatenated copies (fidelity item 1 below), so the cached
> files are not directly runnable references.

---

## Methodology Registry Entry

*Formatted to match docs/methodology/REGISTRY.md structure. Heading levels and labels align with existing entries - copy the `## DurationDiD` section into the "Advanced Estimators" category of the registry once the estimator ships (rename to the shipped class name). It is NOT a "Modern Staggered" entry: the main estimator is two-group with a single common treatment date; the staggered extension (Appendix A.3) is an appendix sketch with an acknowledged bias caveat and no inference.*

## DurationDiD (prospective; name provisional)

**Primary source:** Deaner, B., & Ku, H. (2026). Causal Duration Analysis with Diff-in-Diff. arXiv:2405.05220v2 (numbering below follows that version; unpublished preprint under revise-and-resubmit at *Quantitative Economics* - see the status note above).

The paper studies difference-in-differences when the outcome `Y_{i,t}` is a binary indicator
that individual `i` has entered an **absorbing state** by time `t` (exited unemployment,
passed an exam, left a marriage, had parole revoked; abstract, p. 1). In the paper's own
words the parallel trends assumption on mean outcomes "generally fails in such settings"
(abstract, p. 1; p. 2): if "a sufficiently large share of individuals eventually exit
unemployment, then mean outcomes will tend to converge over time, even absent any treatment
effect" (p. 2), with the explicit exception of footnote 1 (p. 2), "the special case in which
mean outcomes are identical between the two groups". Group means are bounded by one and
increasing, so a fixed mean gap caps the share that can ever be absorbed at `1 + c` and
`1 - c` and forces survivors' exit rates to diverge (Section 1.2, p. 8; the formal ratio
`h^{(0)}_1/h^{(0)}_2 = (1 - E[Y^{(0)}|G=2]) / (1 - E[Y^{(0)}|G=2] - c)`, which grows without
bound, Appendix A.1, p. 32). The bias is systematic: standard DiD attributes the mechanical
narrowing to treatment (p. 2); in the paper's simulation model the DiD bias is "roughly four
times the value of the true treatment effect" with the wrong sign (p. 9).

The proposed fix imposes DiD-type restrictions on group-specific **counterfactual hazard
rates** instead: **common dynamics** (equal changes in hazards over any interval, `h^{(0)}_1(t)
= h^{(0)}_2(t) + c`, Eqs. 2.1/2.3), **proportional hazards** (equal proportional changes,
`h^{(0)}_1(t) = c h^{(0)}_2(t)` with strictly positive hazards, Eqs. 2.2/2.4), or a **general
linear restriction** `h^{(0)}_1(t) = β_1 + Σ_k β_k h^{(0)}_k(t)` across `K - 1` untreated
groups (Eq. 2.5) that nests duration analogues of triple differences and synthetic control
(p. 10). Because the integrated hazard equals the negative log survival, these restrictions
translate into linear restrictions on the estimable **time-average hazard**
`Δ_{t-1}R_{k,t}/(t-1)` with `R_{k,t} = -ln(1 - E[Y_{i,t} | G_i = k])` (Eqs. 2.6-2.12,
Theorems 1-2), so estimation is a plug-in on group-period means with no hazard model fitted.
The method is nonparametric only in that sense: "We sidestep any estimation of the hazard
rates themselves. Our methods are non-parametric in that we avoid specifying a parametric
model for group-specific hazard rates" (p. 3); the restrictions on hazard dynamics are
explicit identifying assumptions (Section 2). Remark 4 (pp. 14-15) shows the common-dynamics
and proportional-hazards conditions are two-way fixed-effects models in `R` with
group-specific linear trends or multiplicative group-time effects, i.e. the exponential model
of Wooldridge (2023) with heterogeneous trends, here derived from the absorbing-state
structure rather than a distributional assumption.

Section 2.2 adds covariate adjustment (covariate-conditional restrictions, and a covariate
balancing re-weighting among period-1 survivors, Theorem 3); Section 2.3 discusses the choice
of specification, spell start dates and right-censoring (Kaplan-Meier-adjusted survival as the
input); Section 3 gives the plug-in estimators (Eqs. 3.1-3.8); Section 3.2 and Appendix B give
an individual-level block bootstrap with pointwise and uniform bands (Algorithm 1) and a
bootstrap pre-treatment specification test (Algorithm 2); Section 3.3 states GMM-type
asymptotic validity without a formal theorem; Section 4 applies the method to the 1989
Austrian unemployment-benefit extension studied by Lalive et al. (2006); Appendix A.3 sketches
staggered adoption, Appendix A.4 a semiparametric covariate approach, Appendix C a Monte Carlo
study (Tables 1-2), Appendix D the proportional-hazards application figures, and Appendix E
the proofs. The authors ship an unlicensed Stata command and MATLAB function (footnote on
p. 1; fidelity notes below).

**Maintainer-approved first-estimator scope (2026-09-05, via the coordinator):** the core
two-group `DurationDiD` with common treatment timing, both the common-dynamics and
proportional-hazards specifications, bootstrap inference and pre-treatment diagnostics.
Covariate adjustment and staggered adoption are deferred from the first estimator. This
review nevertheless transcribes every part of the paper, including the deferred extensions.

**Key implementation requirements:**

*Setting, notation and target (Section 1, pp. 5-7):*
- Units `i = 1..n` observed at times `t = 1..T`; `Y_{i,t}` = 1 if `i`'s spell has ended by
  `t`; spells may already have ended at `t = 1`, none begin after `t = 1` (p. 5).
- `G_i` = group; "group membership is constant over time" (p. 5); Group 1 is the unique
  treated group (p. 9); groups `k = 2..K` are untreated.
- Timing: "An intervention occurs at some time strictly after t* and impacts only
  individuals in Group 1" (p. 5). So `t*` is the LAST pre-treatment period; `t ≤ t*` is
  pre-treatment, `t > t*` post-treatment.
- Observation time vs calendar date: `t(i) = t + s_i` when `t` is spell time and `s_i` the
  spell start date (unemployment application); `t(i) = t` for a fixed-date reform (pp. 5-6).
- Counterfactual `Y^{(0)}_{i,t}` = outcome under no intervention on Group 1 (a group-level
  counterfactual, footnote 2, p. 6). Target:

```
τ_t := E[ Y_{i,t} - Y^{(0)}_{i,t} | G_i = 1 ]                                (unnumbered, p. 6)
```

- Duration reading: `1 - E[Y^{(0)}_{i,t} | G_i = k]` is the counterfactual survival function,
  individuals with `Y^{(0)}_{i,t} = 0` are survivors; if all spells begin at `t = 1`,
  `E[Y^{(0)}_{i,t} | G_i = 1]` is the counterfactual duration CDF at `t` (p. 6).

*Assumption checks / warnings:*
- **Assumption 1 (Absorbing State), p. 6, verbatim:** "`Y_{i,t}` is a binary random variable
  and `Y_{i,t} = 1` implies `Y_{i,s} = 1` for all `t ≤ s`. The same holds for the potential
  outcomes `Y^{(0)}_{i,t}`." → validate binary outcome and within-unit monotonicity; fail
  closed.
- **Assumption 2 (No Anticipation/Spill-Overs), p. 6, verbatim:** "i. For all `t ≤ t*`,
  `Y_{i,t} = Y^{(0)}_{i,t}`, ii. For all `t* < t`, if `G_i ≠ 1` then `Y_{i,t} = Y^{(0)}_{i,t}`."
  Within-group spill-overs are NOT ruled out (p. 7). No-anticipation was introduced in causal
  duration analysis by Abbring & van den Berg (2003) (p. 7). Not testable; document. The
  application shifts `t*` one week earlier to absorb anticipation (p. 25; residual
  anticipation biases estimates toward zero).
- **Common dynamics (2.1)/(2.3)** or **proportional hazards (2.2)/(2.4)**, or the general
  linear restriction (2.5); footnote 4 (p. 10): the condition "is only required to hold
  between periods 1 and T rather than indefinitely far into the past and or future and thus
  leaving initial survivals `E[1 - Y_{i,1} | G_i = k]` unrestricted" (as printed). Under
  proportional hazards the paper requires counterfactual hazards to be "strictly positive"
  (p. 10), so the PH constant satisfies `c > 0`.
- **Data minimum:** "we require at least two periods of pre-treatment data and hence at
  least three periods of data in all. This deviates from standard difference in differences
  which requires only two periods of data. An exception is the case of equal hazard rates
  (`c = 0` in (2.3) or `c = 1` in (2.4)), in which case levels of `R^{(0)}_{k,t}` exhibit
  parallel trends" (p. 12). Theorem 1's premise: "data from at least two pre-treatment
  periods" (p. 13).
- **PH identification (footnote 5, p. 13):** the condition `E[Y_{i,t}|G_i = 2] ≠
  E[Y_{i,1}|G_i = 2]` for some `t ≤ t*` "ensures that `Δ_{t-1}R_{2,t} ≠ 0` for at least one
  value of `1 < t ≤ t*` and so `Δ_{t-1}R_{1,t} = cΔ_{t-1}R_{2,t}` can be inverted to get `c`" -
  ONE nonzero pre-treatment control difference suffices for identification.
- **Over-identification (Remark 1, p. 13):** if `t* > 2`, (2.13) and (2.15) each give
  `t* - 1` equations for `c`; this is what the pre-treatment test exploits.
- **General restriction (Theorem 2, p. 16):** identification needs a unique solution within
  the constraint set `𝓑`; unconstrained, a necessary condition is `t* > K`, and uniqueness is
  "generic whenever this holds".
- **Section 3.3 caveats (pp. 23-24):** (1) nuisance coefficients `{β_k}` must be uniquely
  identified within `𝓑`; (2) if an inequality constraint binds, the estimator "may not be
  differentiable functions of mean outcomes", asymptotic normality fails and "the bootstrap
  does not have correct coverage" (Fang 2019 alternatives); (3) with covariate balancing the
  weights must be bounded above, i.e. propensity scores bounded away from zero (Khan & Tamer
  2010). Asymptotics are `n → ∞` with `K` and `T` fixed.

*Why mean parallel trends generally fails (Sections 1.1-1.2, pp. 7-9; Appendix A.1, p. 32):*

```
E[Y^{(0)}_{i,t} | G_i = 1] = E[Y^{(0)}_{i,t} | G_i = 2] + c,        ∀ 1 ≤ t ≤ T        (1.1)  p. 7
E[Y^{(0)}_{i,t} | G_i = 1] = β_1 + Σ_{k=2}^{K} β_k E[Y^{(0)}_{i,t} | G_i = k]            (1.2)  p. 7
```

- (1.2) is the general class: DiD fixes `β_2 = 1` (`c = β_1`); triple differences (Gruber
  1994) is `K = 4, β_2 = β_3 = 1, β_4 = -1`, `β_1` free (footnote 3, p. 7, gives the two
  equivalent displays); synthetic control (Abadie et al. 2015) is `β_1 = 0` with weakly
  positive coefficients (p. 7; the hazard version on p. 10 adds "sum to 1" - stated only
  there).
- Ceiling argument (p. 8): with 80% / 60% employed pre-treatment, no more than 80% of Group 2
  can ever exit; the counterfactual absorbed share cannot exceed `1 + c` (Group 1) and `1 - c`
  (Group 2); unemployed shares 20% vs 40% force Group 1 survivors to exit at twice the rate,
  later (90% / 70%) at three times the rate. Figure 1.1 (p. 8): panel (a) mean outcomes
  converge, panel (b) the time-average hazards are parallel.

*Identification (Section 2, pp. 9-16):*

```
h^{(0)}_k(t) := lim_{δ↓0} P(Y^{(0)}_{i,t+δ} = 1 | Y^{(0)}_{i,t} = 0, G_i = k) / δ          (unnumbered, p. 9)

h^{(0)}_1(t) - h^{(0)}_1(s) = h^{(0)}_2(t) - h^{(0)}_2(s),   ∀ 1 ≤ s ≤ t ≤ T   (common dynamics)      (2.1)  p. 9
h^{(0)}_1(t) / h^{(0)}_1(s) = h^{(0)}_2(t) / h^{(0)}_2(s),   ∀ 1 ≤ s ≤ t ≤ T   (proportional hazards) (2.2)  p. 10
h^{(0)}_1(t) = h^{(0)}_2(t) + c,          ∀ 1 ≤ t ≤ T                                        (2.3)  p. 10
h^{(0)}_1(t) = c h^{(0)}_2(t),            ∀ 1 ≤ t ≤ T                                        (2.4)  p. 10
h^{(0)}_1(t) = β_1 + Σ_{k=2}^{K} β_k h^{(0)}_k(t),   ∀ 1 ≤ t ≤ T                             (2.5)  p. 10
```

- Special cases of (2.5) (p. 10): common dynamics = `K = 2, β_2 = 1, β_1 = c`; proportional
  hazards = `K = 2, β_1 = 0, β_2 = c`; triple differences in hazards
  `[h^{(0)}_1 - h^{(0)}_2] - [h^{(0)}_3 - h^{(0)}_4] = c` = `β_2 = β_3 = 1, β_4 = -1, β_1 = c`;
  synthetic control = `β_1 = 0, β_k ≥ 0, Σ_{k=2}^{K} β_k = 1`. If hazards are identical across
  groups both (2.3) and (2.4) hold, and mean outcomes may still differ through initial shares
  (p. 10).

```
R^{(0)}_{k,t} := -ln(1 - E[Y^{(0)}_{i,t} | G_i = k])                                          (2.6)  p. 11
R_{k,t}       := -ln(1 - E[Y_{i,t} | G_i = k])                                                (2.7)  p. 11
ΔR^{(0)}_{k,t} := R^{(0)}_{k,t} - R^{(0)}_{k,t-1} = ∫_{t-1}^{t} h^{(0)}_k(r) dr,  1 < t ≤ T   (2.8)  p. 11
ΔR^{(0)}_{1,t} - ΔR^{(0)}_{1,t-1} = ΔR^{(0)}_{2,t} - ΔR^{(0)}_{2,t-1}                        (2.9)  p. 11
ΔR^{(0)}_{1,t} / ΔR^{(0)}_{1,t-1} = ΔR^{(0)}_{2,t} / ΔR^{(0)}_{2,t-1}                        (2.10) p. 11
```

- (2.9) is stated "for any whole number t so that 1 < t ≤ T"; (2.10) has no printed range.
  As printed, at `t = 2` both reference `ΔR_{k,1}`, which (2.8) does not define, so the
  adjacent-difference characterisations effectively start at `t = 3`. The long-difference
  forms below are the ones used for identification and estimation, and the paper's stated
  consequence (two pre-periods, three periods total, p. 12) is unaffected.
- Worked `T = 3` example (p. 12): `E[Y^{(0)}_{i,3}|G=1] = 1 - exp(-R^{(0)}_{1,3})` with
  `R^{(0)}_{1,3} = R_{1,2} + ΔR_{1,2} + ΔR_{2,3} - ΔR_{2,2}` (common dynamics) or
  `R^{(0)}_{1,3} = R_{1,2} + ΔR_{1,2} ΔR_{2,3} / ΔR_{2,2}` (proportional hazards).

```
Δ_s R_{k,t} := R_{k,t} - R_{k,t-s}  (long difference; "time-average hazard" = Δ_{t-1}R_{k,t}/(t-1))   p. 12
Δ_{t-1}R^{(0)}_{1,t} / (t-1) = c + Δ_{t-1}R^{(0)}_{2,t} / (t-1),   ∀ 1 < t ≤ T   (holds for real t ∈ (1, T])   (2.11) p. 12
Δ_{t-1}R_{1,t} = c Δ_{t-1}R_{2,t}                                  (printed without (0) superscripts)     (2.12) p. 12
```

**Theorem 1 (p. 13, verbatim statement):** "Suppose Assumptions 1 and 2 hold and there is
data from at least two pre-treatment periods. Under the common dynamics condition (2.3), `c`
and `R^{(0)}_{1,t}` are identified by,"

```
c = Δ_{t-1}R_{1,t}/(t-1) - Δ_{t-1}R_{2,t}/(t-1),   ∀ 1 < t ≤ t*                              (2.13)
R^{(0)}_{1,t} = R_{1,1} + Δ_{t-1}R_{2,t} + (t-1) c,   ∀ t ≤ T                                (2.14)
```

"If proportional hazards (2.4) holds instead of (2.3) and if `E[Y_{i,t}|G_i = 2] ≠
E[Y_{i,1}|G_i = 2]` for some `t ≤ t*` [footnote 5], then `c` and `R^{(0)}_{1,t}` are
identified by"

```
Δ_{t-1}R_{1,t} = c Δ_{t-1}R_{2,t},   ∀ 1 < t ≤ t*                                            (2.15)
R^{(0)}_{1,t} = R_{1,1} + c Δ_{t-1}R_{2,t},   ∀ t ≤ T                                        (2.16)
```

"In either case, the counterfactual mean outcome is then identified by"

```
E[Y^{(0)}_{i,t} | G_i = 1] = 1 - exp(-R^{(0)}_{1,t})                                          (2.17)
```

- **Remark 1 (Over-Identification, p. 13):** quoted above.
- **Remark 2 (Equivalent Characterizations, p. 14):** (2.13) is equivalent to either
  `c = ΔR_{1,t} - ΔR_{2,t}, ∀ 1 < t ≤ t*` or `c = (Δ_t R_{1,t*} - Δ_t R_{2,t*})/t, ∀ 1 ≤ t ≤
  t*-1`; the theorem's scaled long-difference form is preferred because it generalises to
  uneven increments and because `Δ_{t-1}R̂_{k,t}/(t-1)` "are typically less noisy than those
  of `ΔR_{k,t}` (the former is the average of `ΔR_{k,s}` over `s = 2, ..., t`)".
- **Remark 3 (Uneven Increments and Repeated Cross-Sections, p. 14):** observation times may
  be `t ∈ 𝒯`, "some finite set of real numbers whose smallest element is 1", with (2.13) and
  (2.15) holding for each `t ∈ 𝒯 \ 1` (as printed); and "These objects [group-specific
  means] are identified and estimable with data on repeated cross sections rather than panel
  data" - a statement about point identification/estimation only (the bootstrap of Appendix B
  resamples individual histories).
- **Remark 4 (Comparison to Wooldridge 2023, pp. 14-15):**

```
R^{(0)}_{k,t} = ξ_k - c_k · t + γ_t     (c_1 = 0; TWFE with group-specific linear trends)     (2.18) p. 14
R^{(0)}_{k,t} = ξ_k + c_k · γ_t         (c_1 = 1; multiplicative group-time effects)          (2.19) p. 14
R^{(0)}_{k,t} = ξ_k + γ_t               (c_2 = 0 in (2.18) or c_2 = 1 in (2.19))              (2.20) p. 14
1 - E[Y^{(0)}_{i,t} | G_i = k] = exp(-ξ_k - γ_t)                                    (unnumbered, p. 14)
```

  The last display "is precisely the exponential model given by equations (2.11) and (2.12)
  in Wooldridge (2023) (albeit with `1 - Y_{i,t}` in place of `Y_{i,t}`)"; heterogeneous
  linear trends (Wooldridge 2023, Section 4.2) give (2.18), multiplicative group-time effects
  give (2.19); footnote 6 (p. 15) thanks a referee for the connection. The exponential
  transformation "arises from the duration structure of the data ... not from a distributional
  assumption" (p. 15).

**General linear restrictions (Section 2.1.2, pp. 15-16).** The population analogue
`Δ_{t-1}R^{(0)}_{1,t}/(t-1) = β_1 + Σ_{k=2}^{K} β_k Δ_{t-1}R^{(0)}_{k,t}/(t-1), ∀ 1 < t ≤ T`
(unnumbered, p. 15). **Theorem 2 (statement and (2.21) on p. 15; conclusion continues on
p. 16):** "Suppose Assumptions 1 and 2 hold. In addition suppose that (2.5) holds with
`(β_1, ..., β_K) ∈ 𝓑` for a known set `𝓑`. Then `β_1, ..., β_K` satisfy the equations"

```
Δ_{t-1}R_{1,t}/(t-1) = β_1 + Σ_{k=2}^{K} β_k Δ_{t-1}R_{k,t}/(t-1),   ∀ 1 < t ≤ t*            (2.21) p. 15
R^{(0)}_{1,t} = R_{1,1} + (t-1) β_1 + Σ_{k=2}^{K} β_k Δ_{t-1}R_{k,t},   1 < t ≤ T   (inline, p. 16)
```

  "If the solution to the equations is unique within `𝓑` (i.e., the equations identify
  `β_1, ..., β_K`), then for `1 < t ≤ T`, `ΔR^{(0)}_{1,t}` is identified by [the level formula
  above] and the counterfactual mean outcome by (2.17)" (p. 16; the sentence names the
  first difference but defines the level - as printed). Uniqueness discussion (p. 16): quoted
  under assumption checks.

*Covariates (Section 2.2, pp. 16-19; Appendix A.2, pp. 32-33) - deferred from the first estimator, transcribed in full:*
- `X_i` is a vector of **time-invariant** characteristics (p. 16). Conditional hazard
  `h^{(0)}_k(t; x) = lim_{δ↓0} P(Y^{(0)}_{i,t+δ} = 1 | Y^{(0)}_{i,t} = 0, G_i = k, X_i = x)/δ`
  (p. 16).

```
h^{(0)}_1(t; x) = h^{(0)}_2(t; x) + c(x)                                                      (2.22) p. 16
h^{(0)}_1(t; x) = c(x) h^{(0)}_2(t; x)                                                        (2.23) p. 16
h^{(0)}_1(t, x) = β_1(x) + Σ_{k=2}^{K} β_k(x) h^{(0)}_k(t, x)                                (2.24) p. 16
R^{(0)}_{k,t}(x) := -ln(1 - E[Y^{(0)}_{i,t} | G_i = k, X_i = x]);  R_{k,t}(x) analogously   (p. 17)
E[Y^{(0)}_{i,t} | G_i = 1, X_i = x] = 1 - exp(-R^{(0)}_{1,t}(x))                             (p. 17)
```

- Stratum-wise plug-in (Theorems 1-2 within each `X = x` stratum; formal statement =
  **Theorem A.1 (p. 33)** with (A.1) the covariate-specific restriction and (A.2) the
  covariate-specific (2.21); identification formula `R^{(0)}_{1,t}(x) = R^{(0)}_{1,1}(x) +
  (t-1)β_1 + Σ β_k Δ_{t-1}R_{k,t}(x)`) "may be infeasible if sample sizes are insufficiently
  large" (p. 17).
- **Covariate balancing weights (Section 2.2.1, pp. 17-19)**, balancing "among those who have
  not yet reached the absorbing state in the initial period":

```
ω_k(x) := P(X_i = x | Y_{i,1} = 0, G_i = 1) / P(X_i = x | Y_{i,1} = 0, G_i = k)               (2.25) p. 17
p_k(x) := P(G_i = k | Y_{i,1} = 0, X_i = x)                                                   (p. 18)
ω_k(x) = p_1(x) P(G_i = k | Y_{i,1} = 0) / ( p_k(x) P(G_i = 1 | Y_{i,1} = 0) )   (needs p_k(x) > 0)   (2.26) p. 18
h̃_k(t) := lim_{δ↓0} ( E[ω_k(X_i) Y_{i,t+δ} | G_i = k] - E[ω_k(X_i) Y_{i,t} | G_i = k] ) / ( δ E[ω_k(X_i)(1 - Y_{i,t}) | G_i = k] )   (2.27) p. 18
h̃^{(0)}_1(t) = h̃^{(0)}_2(t) + c   and   h̃^{(0)}_1(t) = c h̃^{(0)}_2(t),   ∀ 1 ≤ t ≤ T         (2.28) p. 18
h̃^{(0)}_1(t) = β_1 + Σ_{k=2}^{K} β_k h̃^{(0)}_k(t),   ∀ 1 ≤ t ≤ T                             (2.29) p. 18
R̃_{k,t} := -ln( E[ω_k(X_i)(1 - Y_{i,t}) | G_i = k] )      (ω_1 ≡ 1, so R̃_{1,t} = R_{1,t})      (p. 19)
```

- Justifications (pp. 18-19): the weighted restrictions may be assumed directly and
  falsified by specification tests; or, **Proposition 1 (p. 19):** "Suppose Assumption 1
  holds and (2.22) holds with `c(x)` equal to a constant `c`. Then `h̃^{(0)}_1(t) =
  h̃^{(0)}_2(t) + c`." Sufficient conditions for constant `c(x)`: equal conditional hazards,
  which holds under conditional ignorability among period-1 survivors
  `Y^{(0)}_{i,t} ⊥ G_i | X_i, Y_{i,1} = 0`; or (2.22) plus an Aalen (1989) additive hazard
  `h_k(t, x) = h̄_k(t) + q(t, x)` (p. 19). No proportional-hazards analogue of Proposition 1
  is stated.
- **Theorem 3 (p. 19):** with `p_k(x) > 0` for all `k` and `x`, and (2.29) with
  `(β_1..β_K) ∈ 𝓑`,

```
Δ_{t-1}R_{1,t}/(t-1) = β_1 + Σ_{k=2}^{K} β_k Δ_{t-1}R̃_{k,t}/(t-1),   ∀ 1 < t ≤ t*            (2.30) p. 19
R^{(0)}_{1,t} = R_{1,1} + (t-1)β_1 + Σ_{k=2}^{K} β_k Δ_{t-1}R̃_{k,t};   E[Y^{(0)}_{i,t}|G_i=1] = 1 - exp(-R^{(0)}_{1,t})
```

*Practical considerations (Section 2.3, pp. 19-20):*
- Equal hazards (`c = 0` in (2.3)) satisfy both specifications (pp. 19-20; **footnote 7**,
  p. 20, is the referee acknowledgement attached to that sentence).
- **Main text, p. 20 (no footnote):** "One advantage of the proportional hazard
  specification over common dynamics is that the latter may imply negative hazard functions,
  yet the definition of these objects implies they are weakly positive. This is more likely
  to arise if the hazard rates are close to zero and differ substantially across groups."
  → diagnostic: an imputed `Δ_{t-1}R̂^{(0)}_{1,t}` that decreases in `t` is an implied
  negative counterfactual hazard.
- Control for the spell start date when `t` is spell time (seasonality) and possibly when
  `t = t(i)` (marriage-duration example) (p. 20).
- **Right-censoring (p. 20):** with `C_{i,t}` = left the study before `t`, one observes
  `Ỹ_{i,t} = (1 - C_{i,t}) Y_{i,t}`; under random censoring (independent of `Y` given `G`)
  "the Kaplan-Meier (Kaplan & Meier (1958)) estimator of the survival function is standard",
  otherwise see Klein & Moeschberger (2003); "One can simply use the corresponding
  censoring-adjusted estimates of `1 - E[Y_{i,t}|G_i = k]` when applying our approach."
  (The lowercase censoring indicator `c_i` collides with the constant `c`, as printed.)

*Estimator equations (Section 3, pp. 20-22, as printed):*

```
R̂_{k,t} := -ln(1 - Ȳ_{k,t})                                                                 (3.1)  p. 20
```

- "the weights `α_t` are positive and `Σ_{t=2}^{t*} α_t = 1`" (p. 21), then: "In our
  empirical application, we set `α = 0` for early observations and otherwise use equal
  weights. More generally, they may be chosen either to (a) place greater emphasis on those
  periods that are closer to the intervention, or (b) minimize the asymptotic variance of the
  estimates" (p. 21). As printed "positive" and "`α = 0`" conflict; the implementable domain
  is non-negative weights with positive total mass (see the tuning table).

```
ĉ = Σ_{t=2}^{t*} α_t ( Δ_{t-1}R̂_{1,t}/(t-1) - Δ_{t-1}R̂_{2,t}/(t-1) )                         (3.2)  p. 21
R̂^{(0)}_{1,t} = R̂_{1,1} + Δ_{t-1}R̂_{2,t} + (t-1) ĉ                                           (3.3)  p. 21
τ̂_t = Ȳ_{1,t} - 1 + exp(-R̂^{(0)}_{1,t})                                                     (3.4)  p. 21
```

- Proportional hazards, **as printed** (p. 21):

```
ĉ = Σ_{t=2}^{t*} α_t Δ_{t-1}R̂_{1,t} Δ_{t-1}R̂_{2,t}  /  Σ_{t=2}^{t*} α_t Δ_{t-1}R̂²_{1,t},
R̂^{(0)}_{1,t} = R_{1,1} + ĉ Δ_{t-1}R̂_{2,t}                                                   (3.5)  p. 21
```

  As-printed notes: the squared term in the denominator carries the subscript `1,t`
  (treated group) and `R_{1,1}` has no hat. Writing `A_t = Δ_{t-1}R̂_{1,t}` and
  `B_t = Δ_{t-1}R̂_{2,t}`, an unscaled weighted least-squares slope of `A_t` on `B_t` through
  the origin is `Σ α_t A_t B_t / Σ α_t B_t²` (the printed denominator subscript corrected);
  under exact proportional hazards the printed form equals `1/c`. The text says "The
  estimators above are both special cases of the procedure below" (p. 21), but the `β_1 = 0`,
  `K = 2` case of (3.6) minimises `Σ α_t (A_t/(t-1) - β_2 B_t/(t-1))²`, whose first-order
  condition is the **scaled** slope `Σ α_t A_t B_t/(t-1)² / Σ α_t B_t²/(t-1)²`; it coincides
  with the unscaled corrected slope only when `(t-1)` is constant over the weighted window
  or the weights absorb `(t-1)^{-2}`. The "special case" claim holds exactly for (3.2)
  ((3.6) with `β_2 = 1` and `Σ α_t = 1` reproduces it) but not for (3.5) as printed. The
  reference code uses none of these: it averages period-wise ratios (fidelity item 4). The
  unscaled corrected slope, the scaled slope and the mean of ratios are consistent for `c`
  under exact proportional hazards and differ in finite samples only through horizon
  weighting; the as-printed form converges to `1/c` and is not implementable. The choice is
  a tracked estimator-PR decision (decision 1 below).

```
{β̂_k}_{k=1}^{K} = argmin_{{β_k} ∈ 𝓑} Σ_{t=1}^{t*} α_t ( Δ_{t-1}R̂_{1,t}/(t-1) - β_1 - Σ_{k=2}^{K} β_k Δ_{t-1}R̂_{k,t}/(t-1) )²   (3.6) p. 21
R̂^{(0)}_{1,t} = R̂_{1,1} + (t-1) β̂_1 + Σ_{k=2}^{K} β̂_k Δ_{t-1}R̂_{k,t}                         (3.7)  p. 21
```

  As printed the sum in (3.6) starts at `t = 1`, where the summand is `0/0`; every other sum
  starts at `t = 2`. "The constraint set `𝓑` can incorporate restrictions like positivity of
  the weights, or that the intercept `β_1` is equal to zero" (p. 21). The prose says
  "regressing `ΔR̂_{1,t}`" while (3.6) regresses the scaled long difference.

- Covariate adjustment (Section 3.1, p. 22; deferred): discrete covariates -

```
ω̂_k(x) = n_k (1 - Ȳ_{k,1}) Σ_i 1{G_i = 1} 1{X_i = x} (1 - Y_{i,1})  /  ( n_1 (1 - Ȳ_{1,1}) Σ_i 1{G_i = k} 1{X_i = x} (1 - Y_{i,1}) )   (3.8) p. 22
```

  defined for all `x` with at least one Group-`k` period-1 survivor; continuous covariates -
  a logit of `1{G_i = k}` on `X_i` among `Y_{i,1} = 0`, then `ω̂_k(x) = n_k p̂_1(x)(1 - Ȳ_{k,1}) /
  (n_1 p̂_k(x)(1 - Ȳ_{1,1}))` (unnumbered, p. 22); weighted negative log survival
  `R̂_{k,t} := -ln( (1/n_k) Σ_i 1{G_i = k} ω̂_k(X_i)(1 - Y_{i,t}) )` (unnumbered, p. 22)
  substituted into (3.2)-(3.7). Weight estimation must be repeated inside every bootstrap
  draw (p. 23).

*Standard errors and specification test (Section 3.2, pp. 22-23; Appendix B, pp. 38-40):*
- Default: **individual-level block bootstrap** - "one independently resamples individuals
  uniformly with replacement and forms a new sample using the complete series of outcomes
  and covariates for each individual resampled" (p. 38); motivated by Bertrand et al. (2004);
  no stratification by group is mentioned; no clustering above the individual.
- SE: "The standard deviation `σ̂_t` of `τ̂*_{b,t}` over the bootstrap samples `b = 1, ..., B`
  is taken as the standard error for `τ̂_t`" (p. 38).
- Pointwise band: `q̂_{1-α,t}` = the `1 - α` quantile of `|τ̂*_{b,t} - τ̂_t| / σ̂_t`;
  `CI_{1-α,t} = [τ̂_t - q̂ σ̂_t, τ̂_t + q̂ σ̂_t]` (p. 38; symmetric studentised, NOT a percentile
  interval; the display drops the `t` subscript on `q̂`).
- Uniform band: replace `q̂_{1-α,t}` by the `1 - α` quantile over `b` of
  `max_{t* ≤ s ≤ T} |τ̂*_{b,s} - τ̂_s| / σ̂_s` (p. 38).
- Alternative: none analytic. Section 3.3: all procedures "can be written as generalized
  method of moments estimators or sequential generalized method of moments estimators", so
  consistency and bootstrap coverage follow from standard regularity conditions; "following
  the example of Wooldridge (2023), we omit a formal statement here" (p. 23).
- Clustering: not discussed (individual histories independent, p. 22).
- **Specification test, Section 3.2 (pp. 22-23), as printed:** for `t = 2, ..., t*`,

```
δ_t = ( Δ_{t-1}R_{1,t}/(t-1) - Δ_{t-1}R_{2,t}/(t-1) ) - ( Δ_{t-1}R_{1,t*}/(t-1) - Δ_{t-1}R_{2,t*}/(t-1) ) = 0     (CD, p. 23)
δ_t = Δ_{t-1}R_{1,t}/Δ_{t-1}R_{2,t} - Δ_{t-1}R_{1,t*}/Δ_{t-1}R_{2,t*} = 0                                 (PH, p. 23)
```

  Note the `t*` reference terms are printed with `Δ_{t-1}` and denominator `(t-1)`, whereas
  **Algorithm 2 step 1 (p. 40)** prints them as `Δ_{t*-1}R̂_{k,t*}/(t*-1)` - the time-average
  hazard at `t*`. The reference code implements the Algorithm 2 form (`durationdid.ado:348`
  for common dynamics, `:355` for proportional hazards; `durationDiD.m:47` and `:51`). The
  text says the test "rejects if and only if there is some time `2 ≤ t ≤ t*` for which these
  confidence bands do not contain zero" (p. 23) while Algorithm 2 tests `2 ≤ t ≤ t* - 1`
  (`δ̂_{t*} ≡ 0` by construction). **The paper defines no p-value**; the test is a uniform-band
  test. The application uses a **60% uniform band, i.e. α = 0.40** ("even at the highly
  conservative 60% level (the p-value is 0.643)", p. 28 - the p-value there comes from the
  authors' code, fidelity item 2).

*Algorithm 1 (Block Bootstrap Inference, p. 39, verbatim):*
1. "For each `t = t*, , ..., T` evaluate the estimator `τ̂_t` as in Section 3.1." (double
   comma as printed)
2. "for `b = 1, 2, ..., B` do"
3. "Independently draw a sequence of `n` natural numbers uniformly from `{1, 2, ..., n}`.
   Denote the sequence by `{j_b(1), j_b(2), ..., j_b(n)}`."
4. "For each `t = t*, ..., T` evaluate the estimator `τ̂_t` using `Y_{t,j_b(i)}` in place of
   `Y_{i,t}`, `G_{t,j_b(i)}` in place of `G_{i,t}`, and `X_{j_b(i)}` in place of `X_i` wherever
   they appear in the formula. Call the resulting estimator `τ̂*_{b,t}`." (subscript order
   and the `t` on `G` as printed)
5. "end for"
6. "Calculate bootstrap standard errors `σ̂_t` for `t = t* + 1, ..., T` as the standard
   deviation of the sample `{τ̂*_{b,t}}_{b=1}^{B}`."
7. "For each `t = t* + 1, ..., T` let the pointwise level `1 - α` critical value `q̂_{1-α,t}`
   be the `1 - α` quantile of `{|τ̂*_{b,t} - τ̂_t|/σ̂_t}_{b=1}^{B}`. For uniform critical values,
   instead use the `1 - α` quantile of `{max_{t* ≤ s ≤ T} |τ̂*_{b,s} - τ̂_s|/σ̂_s}_{b=1}^{B}`
   (note this does not depend on `t`)." (the max includes `s = t*`, for which `σ̂_{t*}` is
   never defined in step 6 - as printed)
8. "Form confidence bands by `CI_{1-α,t} = [τ̂_t - q̂_{1-α,t} σ̂_t, τ̂_t + q̂_{1-α,t} σ̂_t]`"

*Algorithm 2 (Specification testing, p. 40, verbatim):*
1. "For each `t = 2, ..., t*` and `k = 1, 2` evaluate the estimate `Δ_{t-1}R̂_{k,t}` with
   formula given in Section 2. Using these estimates evaluate the difference-in-differences
   below to test the common dynamics case"

```
δ̂_t = ( Δ_{t-1}R̂_{1,t}/(t-1) - Δ_{t-1}R̂_{2,t}/(t-1) ) - ( Δ_{t*-1}R̂_{1,t*}/(t*-1) - Δ_{t*-1}R̂_{2,t*}/(t*-1) )
```

   "or for proportional hazards"

```
δ̂_t = Δ_{t-1}R̂_{1,t}/Δ_{t-1}R̂_{2,t} - Δ_{t*-1}R̂_{1,t*}/Δ_{t*-1}R̂_{2,t*}
```

2. "for `b = 1, 2, ..., B` do"
3. (draw `n` indices uniformly with replacement, as in Algorithm 1)
4. "For `k = 1, 2` and each `t = 2, ..., t*` evaluate the estimator `Δ_{t-1}R̂_{k,t}` using
   `Y_{t,j_b(i)}` in place of `Y_{i,t}`, `G_{t,j_b(i)}` in place of `G_{i,t}`, and `X_{j_b(i)}`
   in place of `X_i` wherever they appear in the formula. Call the resulting estimator
   `R̂*_{b,k,t}`. Using these, evaluate the following quantity for common dynamics
   `δ̂*_{b,t} = (Δ_{t-1}R̂*_{b,1,t}/(t-1) - Δ_{t-1}R̂*_{b,2,t}/(t-1)) - (Δ_{t*-1}R̂*_{b,1,t*}/(t*-1)
   - Δ_{t*-1}R̂*_{b,2,t*}/(t*-1))` or for proportional hazards `δ̂*_{b,t} =
   Δ_{t-1}R̂*_{b,1,t}/Δ_{t-1}R̂*_{b,2,t} - Δ_{t*-1}R̂*_{b,1,t*}/Δ_{t*-1}R̂*_{b,2,t*}`."
5. "end for"
6. "Calculate bootstrap standard errors `σ̂_t` for `t = 2, ..., t* - 1` as the standard
   deviation of the sample `{δ̂*_{b,t}}_{b=1}^{B}`."
7. "For each `t = 2, ..., t* - 1` let the pointwise level `1 - α` critical value `q̂_{1-α}` be
   the `1 - α` quantile of `{max_{2 ≤ s ≤ t*-1} |δ̂*_{b,s} - δ̂_s|/σ̂_s}_{b=1}^{B}`." (labelled
   "pointwise" but it is the uniform, sup-t critical value; no `t` subscript)
8. "Form confidence bands by `CI_{1-α,t} = [τ̂_t - q̂_{1-α} σ̂_t, τ̂_t + q̂_{1-α} σ̂_t]`" (written
   around `τ̂_t` rather than `δ̂_t`, as printed)
9. "Reject pre-treatment parallel trends if for any `2 ≤ t ≤ t* - 1`, the interval
   `CI_{1-α,t}` does not contain zero."

- **Scope of the two algorithms.** Algorithm 2's specification test is two-group only
  (`k = 1, 2` in steps 1 and 4); no `K > 2` or staggered version of the test is given.
  Algorithm 1 carries no group index: it is stated generically for "the estimator `τ̂_t`"
  (steps 1 and 4), the Appendix B prose says "one computes the estimate `τ̂_t` using the
  bootstrap sample in place of the original data" (p. 38), and Section 3.3 asserts bootstrap
  coverage for "All of the procedures we have described" (p. 23) - so Algorithm 1 covers the
  K-group estimator (3.6)-(3.7) provided the nuisance coefficients (and, with covariates,
  the weights) are re-estimated inside each draw (p. 23). The PH `δ̂_t` subtracts one
  **shared anchor ratio** `Δ_{t*-1}R̂_{1,t*}/Δ_{t*-1}R̂_{2,t*}` from every tested horizon.

*Extensions transcribed (deferred from the first estimator):*
- **Appendix A.3 Staggered Adoption (pp. 33-35).** `G_i` is redefined as "the last period
  before which the individual is treated" - `G_i = k` means treated some time after `k`,
  `G_i = T` means never treated within the data (p. 33). Restrictions for all `1 ≤ s ≤ t ≤ T`
  and `1 ≤ k, l ≤ T`:

```
h^{(0)}_k(t) - h^{(0)}_k(s) = h^{(0)}_l(t) - h^{(0)}_l(s)                                     (A.3) p. 33
h^{(0)}_k(t)/h^{(0)}_k(s) = h^{(0)}_l(t)/h^{(0)}_l(s)     (with h_k(t) > 0 for all t, k)       (A.4) p. 33
```

  **Assumption 2\* (No Anticipation/Spill-Overs: Staggered Adoption), p. 33:** "For all
  `t ≤ G_i`, `Y_{i,t} = Y^{(0)}_{i,t}`." Under (A.3) there is `c_{t,s}` with
  `Δ_{t-1}R^{(0)}_{k,t}/(t-1) - Δ_{s-1}R^{(0)}_{k,s}/(s-1) = c_{t,s}` for all `k`; under (A.4)
  the ratio equals `c_{t,s}` (pp. 33-34). Observed-data versions hold for `k ≥ t`:

```
Δ_{t-1}R_{k,t}/(t-1) - Δ_{s-1}R_{k,s}/(s-1) = c_{t,s},   ∀ k ≥ t                             (A.5) p. 34
( Δ_{t-1}R_{k,t}/(t-1) ) / ( Δ_{s-1}R_{k,s}/(s-1) ) = c_{t,s},   ∀ k ≥ t   (needs Δ_{s-1}R_{k,s}/(s-1) > 0)   (A.6) p. 34
```

  `c_{t,s}` is identified for any `1 < s ≤ t ≤ T` with `P(G_i ≥ t) > 0`; then for
  `s ≤ k < t` (p. 34, **as printed**): under (A.3) `R^{(0)}_{k,t} = R_{k,1} + (t-1)c_{t,s} +
  (t-1) Δ_{s-1}R_{k,s}/(s-1)`; under (A.4) `R^{(0)}_{k,t} = R_{k,t} + (t-1)c_{t,s}
  Δ_{s-1}R_{k,s}/(s-1)` - the PH display starts from the factual period-`t` value `R_{k,t}`
  (a post-treatment quantity of a treated group), whereas the CD display and BOTH plug-in
  estimators on the same page start from `R_{k,1}`: `R̂^{(0)}_{k,t} = R̂_{k,1} + (t-1)ĉ_{t,s} +
  (t-1)Δ_{s-1}R̂_{k,s}/(s-1)` and `R̂^{(0)}_{k,t} = R̂_{k,1} + (t-1)ĉ_{t,s} Δ_{s-1}R̂_{k,s}/(s-1)`.
  `R_{k,1}` is the baseline consistent with (A.6) and Theorem 1. Plug-in `ĉ_{t,s}` = a
  weighted mean over `k ∈ 𝒢, k ≥ t` of the differences (or ratios) with positive weights
  `α_k` summing to one; "Setting `s = k` is a natural choice" (p. 34). **Caveat (p. 35):**
  with few individuals in group `k` the time-average hazard estimate "is not only noisy in
  general but also biased, because it is a non-linear transformation of the sample mean. As
  such, estimates of `ĉ_{t,s}` are biased" - set `α_k` small for small groups, use other
  methods if all groups are small; bias adjustments are "a topic for future work". No
  inference procedure and no `α_k` recipe are given.
- **Appendix A.4 Semi-parametric covariate adjustment (pp. 35-38).**

```
h^{(0)}_1(t, x) = β_1(x) + Σ_{k=2}^{K} β_k(x) h^{(0)}_k(t, x)                                (A.7) p. 35
h^{(0)}_k(t, x) = φ(x'γ_k) h̄^{(0)}_k(t)      (φ known, strictly positive; group-k Cox model)   (A.8) p. 35
h̄^{(0)}_1(t) = β_1 + Σ_{k=2}^{K} β_k h̄^{(0)}_k(t)        (restriction on BASELINE hazards)     (A.9) p. 35
Δ_{t-1}R̄^{(0)}_{k,t} := Δ_{t-1}R^{(0)}_{k,t}(x) / φ(x'γ_k)        (does not depend on x)         (p. 36)
E[Y^{(0)}_{i,t} | Y^{(0)}_{i,1} = 0, G_i = k, X_i = x] = 1 - exp(-φ(x'γ_k) Δ_{t-1}R̄^{(0)}_{k,t})   (A.10) p. 36
E[Y_{i,t} | Y_{i,1} = 0, G_i = k, X_i = x] = 1 - exp(-φ(x'γ_k) Δ_{t-1}R̄^{(0)}_{k,t})   (1 ≤ t ≤ t* and/or k ≠ 1)   (A.11) p. 36
Δ_{t-1}R̄^{(0)}_{1,t}/(t-1) = β_1 + Σ_{k=2}^{K} β_k Δ_{t-1}R̄^{(0)}_{k,t}/(t-1),   ∀ 1 < t ≤ t*   (A.12) p. 37
E[Y^{(0)}_{i,t} | G_i = 1] = E[Y_{i,1} | G_i = 1] + E[ (1 - Y_{i,1}) (1 - exp(-φ(γ_1' X_i) Δ_{t-1}R̄^{(0)}_{1,t})) ]   (A.13) p. 37
```

  (A.13) as printed leaves the second expectation unconditional; the Theorem 4 proof
  (p. 50) equates `E[Y^{(0)}_{i,t} | G_i = 1, Y_{i,1} = 0] = 1 - E[(1 - Y_{i,1}) exp(-φ(X_i'γ_1)
  Δ_{t-1}R̄^{(0)}_{1,t}) | G_i = 1] / E[1 - Y_{i,1} | G_i = 1]` with `(E[Y^{(0)}_{i,t} | G_i = 1] -
  E[Y_{i,1} | G_i = 1]) / E[1 - Y_{i,1} | G_i = 1]`, so the second expectation must be
  **conditional on `G_i = 1`**: `E[Y^{(0)}_{i,t} | G_i = 1] = E[Y_{i,1} | G_i = 1] +
  E[(1 - Y_{i,1})(1 - exp(-φ(γ_1' X_i) Δ_{t-1}R̄^{(0)}_{1,t})) | G_i = 1]`. The p. 38 empirical
  analogue averages with `1{G_i = 1}` and `1/n_1`, i.e. the Group-1 conditional mean. Without
  the conditioning the pooled average differs from the identified object whenever the
  covariate distribution among period-1 survivors differs across groups.

  Footnote 8 (p. 35): the baseline hazard differs from the unconditional hazard. Footnote 9
  (p. 36): `Δ_s R^{(0)}_{k,t}(x)/φ(x'γ_k) = ∫_{t-s}^{t} h̄^{(0)}_k(r) dr`. Footnote 10 (p. 36):
  with an exponential link and `0` in the support of `X`, the `X = 0` cell inverts to
  `Δ_{t-1}R̄^{(0)}_{k,t}` - "This demonstrates the importance of not including an intercept."
  **Theorem 4 (pp. 36-37):** under Assumptions 1-2, (A.8) with `φ` strictly positive and
  strictly increasing, full support of `X_i` "conditional on `Y_{i,t} = 0` and `G_i = k`" (as
  printed), and (A.9) with `β ∈ 𝓑`, (A.11) holds and identifies `Δ_{t-1}R̄^{(0)}_{k,t}` and
  `γ_k` for all `k > 1` and `1 < t ≤ t*`, and `β` satisfies (A.12); if unique, the text then
  prints `Δ_{t-1}R̄^{(0)}_{1,t} = (1 - t)β_1 + Σ β_k Δ_{t-1}R̄^{(0)}_{k,t}` (p. 37; multiplying
  (A.12) by `(t-1)` gives `(t-1)β_1`, the coefficient Theorem A.1 uses - as printed) and
  (A.13). **A.4.1 estimation (pp. 37-38):** Cox partial likelihood is possible but ties in
  discrete time motivate non-linear least squares; `{γ̂_k}` and `{Δ_{t-1}R̄̂_{k,t}}` minimise
  `Σ_{(k,t)∈𝒯} Σ_i 1{G_i = k}(1 - Y_{i1})(Y_{i,t} - 1 + exp(-φ(x'γ_k) r_{k,t}))²` over pairs
  with `k > 1` and/or `1 < t ≤ t*` (optionally `γ_k = γ`); impute the post-treatment
  `Δ_{t-1}R̄̂^{(0)}_{1,t}` via Theorems 1-2; the empirical ATT analogue (p. 38) is printed as
  `Ȳ_{1,t} - Ȳ_{1,1} - (1/n_1) Σ_{n=1}^{n} (1 - Y_{it}) 1{G_i = 1}(1 - exp(-φ(γ̂_1' X_i)
  Δ_{t-1}R̄̂^{(0)}_{1,t}))` - with the survival indicator `(1 - Y_{it})` (period `t`) where
  (A.13) weights by baseline survival `(1 - Y_{i,1})`, and a summation index `n` (as printed;
  the baseline-survival weight `(1 - Y_{i,1})` of (A.13) is the one implied by the
  identification result, and the analogue's `1{G_i = 1}/n_1` averaging supplies the
  `G_i = 1` conditioning that the printed (A.13) omits). No inference recipe is given for
  A.4.
- **Appendix E proofs (pp. 46-50):** Theorem 1 from Theorem 2 by encoding `β_2 = 1` or
  `β_1 = 0` in `𝓑` (p. 46); Theorem 2 via `R^{(0)}_{k,t} - R^{(0)}_{k,1} = ∫_1^t h^{(0)}_k`,
  (E.1) `Δ_{t-1}R^{(0)}_{k,t}/(t-1) = (1/(t-1))∫_1^t h^{(0)}_k(s)ds`, the affine combination
  (E.2), and Assumption 2.i giving (E.3) on observables (pp. 46-47; the text there states
  "`R^{(0)}_{1,t} = R_{1,t}`" where the following display uses the `t = 1` value `R_{1,1}` -
  as printed); Proposition 1 (pp. 47-48) integrates the constant hazard gap from 1 to `t` yet
  prints the factor `exp(tc)` (propagated through three displays; evaluating at `t = 1` forces
  `exp((t-1)c)`; the discrepancy cancels in the final `-(d/dt) ln` step); Theorem 3 (pp. 48-49)
  via `R̃`; Theorem 4 (pp. 49-50) with (E.4)-(E.6), where the p. 50 display
  `Δ_{t-1}R̄^{(0)}_{k,t} = -ln(E[1 - Y_{i,t} | G_i = k, X_i = 0, Y_{i,1} = 0]) / (-φ(0))` carries
  one minus sign too many relative to (E.6) (which implies `-ln(·)/φ(0)`) and to the
  immediately following `γ_k` display.

*Time grid and minima (counts over the observed grid; Remark 3, p. 14; p. 12; Remark 1, p. 13):*
- Let `𝒯 = {t_1 < t_2 < … < t_m}` be the observed grid (integers or reals; the paper's
  convention normalises the smallest element to 1) and `t*` a grid element at index `j*`.
  Every denominator `(t - 1)` in the paper is elapsed time since the first grid point, i.e.
  `t - t_1`.
- **Estimation minimum:** `j* ≥ 2` (at least two pre-treatment grid points including `t_1`,
  p. 12) and `j* < m` (at least one post-treatment grid point).
- `α` weights are indexed by the `j* - 1` grid points `t_2..t_{j*}` in grid order; a burn-in
  is a grid label (the first grid point with positive weight); an explicit weight vector has
  length `j* - 1`.
- **Diagnostic minimum:** the Algorithm 2 test set is `t_2..t_{j*-1}`; the test is available
  whenever it contains at least one grid point besides the anchor `t_{j*}` (`j* ≥ 3`; at
  `j* = 3` it is a single-point, degenerate-uniform test; Remark 1's over-identification
  holds whenever `t* > 2`). The test set is NOT truncated by the estimation weights (the
  reference code's `burnin..t*` start is an optional, documented deviation).

*Edge cases (detection → handling proposals):*
- **Input validation, fail closed:** empty sample; missing treated or untreated arm; **panel
  completeness over the estimation window** - every unit has exactly one row at every grid
  point from `t_1` through the extrapolation end (a unit observed at only some points is
  indistinguishable from right-censoring, which the paper routes through the deferred
  Kaplan-Meier-adjusted input, p. 20; the in-repo precedent `diff_diff/changes_in_changes.py`
  asserts balance before pivoting histories, lines 734-748); **treatment constant within
  unit** (p. 5); `j* < m`; empty group-period cells; NaN / missing values in the unit, time,
  treatment or outcome columns (`validate_binary` in `diff_diff/utils.py:52-70` strips NaN
  before checking and passes empty arrays, so explicit checks are required); duplicate
  `(unit, time)` rows; non-monotone `Y` within unit (Assumption 1); non-binary treatment;
  **non-numeric time labels** (datetime64, `pd.Period`, strings, categoricals) rejected with a
  message pointing at conversion to elapsed units, because `t - t_1` arithmetic is
  load-bearing; `t*` not in the grid; extrapolation end beyond the data.
- **Exact-zero survival `Ȳ_{k,t} = 1`, role-sensitive, no clipping.** (3.1) maps a fully
  absorbed cell to `R̂ = +∞`; the reference propagates it unguarded (`durationDiD.m:39-40`:
  `log(0) = -Inf`, then `∞ - ∞ = NaN` under CD and `∞/∞` under PH). By Assumption 1 a fully
  absorbed cell stays fully absorbed at every later grid point. Proposed handling:
  (i) baseline `Ȳ_{k,1} = 1` for either group → no period-1 survivors, every `H_k` undefined
  → the fit raises `ValueError` naming the group. (ii) A pre-treatment cell `t_0 ≤ t*` with
  `Ȳ_{k,t_0} = 1` → the summands of (3.2)/(3.5) at every `t ≥ t_0` are undefined and, because
  `t* ≥ t_0`, the Algorithm 2 anchor `Δ_{t*-1}R̂_{k,t*}` is infinite, so the specification
  test is unavailable regardless of weights. `ĉ` itself is NOT necessarily undefined: (3.2)
  and (3.5) reference `t*` only inside the `α`-weighted sum, so a weight vector that puts
  zero weight on every affected period leaves `ĉ` finite. **Product choice (not a
  mathematical necessity):** fail closed by default - raise naming group and period, because
  the default and burn-in weight constructions put positive weight on `t*` and the diagnostic
  is lost; when the user supplies an explicit weight vector that zeroes every affected
  pre-treatment period, estimate `ĉ` from the remaining periods, mark the specification test
  unavailable, warn, and then apply case (iii) to the post-treatment periods (a control cell
  absorbed at `t_0 ≤ t*` is absorbed at every post period too, so (ii) and (iii) describe the
  same dataset consistently). (iii) A post-treatment **control** cell `Ȳ_{2,t} = 1`: under
  common dynamics (3.3) gives `R̂^{(0)}_{1,t} = +∞` for any finite `ĉ`, so `τ̂_t = Ȳ_{1,t} - 1`
  is the well-defined limiting value, reported with a `UserWarning` and a results caveat;
  under proportional hazards (3.5) gives `R̂^{(0)}_{1,t} = R̂_{1,1} + ĉ · ∞`, which is `+∞`
  (same limiting value) only when `ĉ > 0` - the paper's PH domain (strictly positive hazards,
  p. 10). (iv) A post-treatment **treated** cell `Ȳ_{1,t} = 1` needs no `R̂_{1,t}` and is
  ordinary. (v) Inside bootstrap draws the same rules apply, but a draw hitting (i)/(ii) is a
  non-finite replicate handled by the valid-replicate gate, not an exception. No
  undocumented clipping of survival or `R̂` anywhere.
- **Proportional-hazards parameter domain.** The paper assumes strictly positive
  counterfactual hazards under PH (p. 10), so `c = h_1/h_2 > 0`. An in-sample `ĉ ≤ 0` is
  reachable (with monotone `Y` every `Δ_{t-1}R̂_{k,t} ≥ 0`, so the weighted-LS slope and the
  reference's mean of ratios both return 0 when the treated group has no pre-treatment exits
  in the weighted window) and is outside the model: report NaN inference for the PH fit with
  a warning naming the cause; never a silent value. With `ĉ = 0` the PH imputation collapses
  to `R̂^{(0)}_{1,t} = R̂_{1,1}` (`τ̂_t = Ȳ_{1,t} - Ȳ_{1,1}`) and numerically `0 · ∞` is NaN in
  case (iii) (`durationDiD.m:50` reproduces this shape).
- **PH zero denominators, split by object.** *Identification* needs only ONE nonzero
  pre-treatment control difference (footnote 5, p. 13). *Estimation* fails only when the
  aggregate denominator of the chosen `ĉ` is zero (`Σ α_t B_t²` under decision 1(i),
  `Σ α_t B_t²/(t-1)²` under 1(iv)); under the reference's mean of ratios
  (`durationDiD.m:49`, `nanmean(H1./H2)`) the three reachable cases differ: a horizon with
  `H1 > 0` and `H2 = 0` gives `Inf`, which `nanmean` retains and which poisons the mean; a
  horizon with `H1 = H2 = 0` gives `NaN`, which `nanmean` drops, so an otherwise usable
  window still yields a finite `ĉ` (this `0/0` branch is reachable: it is the "treated
  group has no pre-treatment exits" case above); a window with every horizon undefined
  returns `NaN` (the Mata path filters all non-finite ratios through `select(diffs, diffs
  :< .)`, `durationdid.ado:352`). The library policy - reject any zero-denominator horizon
  inside the weighted window with NaN inference and a warning - is a **deliberate
  deviation** from the reference, not the reference's behaviour. *Diagnostics:* the
  PH `δ̂_t` subtracts the shared anchor ratio; validate the anchor denominator
  `Δ_{t*-1}R̂_{2,t*}` separately - if it is zero the whole PH specification test is
  unavailable (NaN test fields, empty `δ̂` arrays, warning) while an otherwise identified fit
  stays intact; horizon-specific `Δ_{t-1}R̂_{2,t} = 0` for `t` in the test set drops only that
  horizon with a warning. The CD statistic has no denominator.
- **Common dynamics implied negative hazard** (p. 20): imputed `Δ_{t-1}R̂^{(0)}_{1,t}`
  decreasing in `t` → non-monotone `E[Y^{(0)}]` → warn.
- **Zero bootstrap SD, per surface.** The reference divides unguarded (`durationDiD.m:118-119`;
  Mata adds `1e-100`, `durationdid.ado:496-498`). Library policy: the stored per-period `se` is
  set to NaN for a zero-spread horizon (the `changes_in_changes.py:1070-1075` precedent) so
  `safe_inference` applies the **joint-NaN inference contract** - the point estimate stays
  available while `se`, `t_stat`, `p_value`, `conf_int` and the affected band endpoints are
  NaN (`diff_diff/utils.py:404-405`); that horizon is excluded from the pointwise-band
  quantile and from the max defining the uniform critical value rather than poisoning the
  whole band; the specification-test statistic likewise excludes zero-SD horizons from the
  test set with a warning; if no valid horizon remains the band or the test is reported
  unavailable (NaN critical value, NaN p-value). The `1e-9` IQR floor of
  `changes_in_changes.py:816-820` is a qte-parity port and is not adopted.
- **Invalid bootstrap draws, two families.** The paper and the reference keep the two
  bootstrap summaries apart (Algorithm 1 summarises only `τ̂*_{b,t}`, Algorithm 2 only
  `δ̂*_{b,t}`; `durationDiD.m:117-119` builds `boot_tau` and `boot_delta` independently).
  Apply the library's valid-replicate gate separately to each family: one finite-row mask,
  `n_valid` and minimum-valid threshold over the Algorithm 1 replicates (dropped jointly
  across post horizons because the uniform band takes a max across them), another over the
  Algorithm 2 replicates; a draw finite for `τ̂` but NaN for `δ̂` (e.g. a PH draw with a zero
  early control difference) stays in the ATT bands. Each family reports `n_valid`, calls
  `warn_bootstrap_failure_rate` (`diff_diff/bootstrap_utils.py:79`), and NaNs its SEs /
  critical values only when `n_valid < max(2, 0.5 · n_bootstrap)`
  (`changes_in_changes.py:241`, gate at `:801-808`).
- **Unavailable specification test:** when `j* < 3`, the anchor denominator is zero (PH),
  or no tested horizon survives, the estimator still fits; results carry
  `spec_test_p_value = NaN`, `spec_test_crit_value = NaN`, empty (length-0) `δ̂` / band arrays,
  a caveat string naming the reason, and a `UserWarning`; never an exception when estimation
  is valid.
- Never-absorbed units: all-zero rows in the long shape; `inf` reserved for never-absorbed
  durations in the cross-section shape (never a sentinel period).
- Deferred-scope cases recorded for later: balancing cells with no treated mass or fewer
  than two untreated units (p. 25 drops such calendar dates; the reference zeroes the cell
  and rescales); extreme balancing weights / propensities near zero (Section 3.3 caveat 3);
  binding inequality constraints in `𝓑` (caveat 2).

*Specification-test p-value (a library extension - the paper defines none):* with test set
`𝒯_test = {t_2..t_{j*-1}}` (minus dropped horizons), `σ̂_t = SD_b(δ̂*_{b,t})` over valid draws,
`T̂ = max_{t∈𝒯_test} |δ̂_t|/σ̂_t` and `T*_b = max_{t∈𝒯_test} |δ̂*_{b,t} - δ̂_t|/σ̂_t` (the same
centred, studentised statistic Algorithm 2 step 7 uses for the band):

```
p = max( (1/n_valid) Σ_b 1{ T*_b ≥ T̂ },  1/(n_valid + 1) )
```

Inclusive comparison and the `1/(n_valid + 1)` floor follow `compute_bootstrap_pvalue`
(`diff_diff/bootstrap_utils.py:278-311`). Reference differences: strict `>`, `abs(max(·))`
instead of `max|·|`, test set `burn_in..t*` including the zero anchor (fidelity item 2).

*Monte Carlo design (Appendix C, pp. 40-44) - future test oracle:*

```
h^{(0)}_1(t) = ( 1 + sqrt(t/T) - (1/2)(t/T - 1/2)² + c ) / (T - 1)
h^{(0)}_2(t) = ( 1 + sqrt(t/T) - (1/2)(t/T - 1/2)² ) / (T - 1)
h_1(t)       = ( 1 + sqrt(t/T) - (1/2)(t/T - 1/2)² + c + β 1{t ≥ t*} ) / (T - 1)            (p. 41)
P(Y_{t+1,i} = 1 | Y_{i,t} = 0, G_i = k) = 1 - exp( -∫_t^{t+1} h_k(s) ds )   (evaluated numerically)   (p. 41)
```

- The prose says the intervention "occurs at time `t*`" and the indicator is `1{t ≥ t*}`
  (p. 41), against the p. 5 convention "strictly after `t*`"; the prose also says the
  switching probability is for "between the discrete intervals `t - 1` and `t`" while the
  display is the `t → t + 1` transition (as printed).
- **Table 1 (p. 41):** `T = 20`, `t* = 11`, `E[Y_{i,1}|G_i = 1] = 0.4`, `E[Y_{i,1}|G_i = 2] = 0.2`,
  `c = 0.5`, `β = 1`. 1,000 simulated datasets; 1,000 bootstrap replications; "We set the
  weights all equal" (p. 42); **confidence bands AND the parallel-trends test at the 95%
  level** (Table 2 footnote, p. 44) - a 5% pre-trend test, unlike the application's 60% band.
  Figures C.1-C.4 (pp. 42-43; the prose cites them as "Figure A.1-A.4").
- **This is a common-dynamics DGP** ("a continuous time duration model in which the common
  dynamics condition holds and we use this same model as the data-generating process in the
  Monte Carlo exercises", p. 13; the hazards differ by the constant `c/(T-1)`, so their ratio
  moves with `t`), with no covariates. It validates the common-dynamics half of the approved
  scope only; the checklist carries a separate proportional-hazards fixture.
- **Table 2 (p. 44), "Simulation Performance", columns `n` | Absolute Bias | Mean-Squared
  Error | Confidence Band Coverage (Uniform) | Confidence Band Coverage (Pointwise) |
  Parallel Trends Test Rejects:**

| Duration Difference-in-Differences: n | Absolute Bias | MSE | Uniform coverage | Pointwise coverage | PT test rejects |
|---:|---:|---:|---:|---:|---:|
| 100 | 0.00333 | 0.00176 | 0.962 | 0.957 | 0.058 |
| 500 | 0.00111 | 0.00034 | 0.949 | 0.951 | 0.041 |
| 1000 | 0.00024 | 0.00017 | 0.955 | 0.953 | 0.051 |
| 5000 | 0.00008 | 0.00003 | 0.945 | 0.946 | 0.058 |
| 10000 | 0.00010 | 0.00002 | 0.960 | 0.956 | 0.052 |

| Standard Difference-in-Differences: n | Absolute Bias | MSE | Uniform coverage | Pointwise coverage | PT test rejects |
|---:|---:|---:|---:|---:|---:|
| 100 | 0.070 | 0.008 | 0.680 | 0.739 | 0.038 |
| 500 | 0.067 | 0.006 | 0.077 | 0.266 | 0.079 |
| 1000 | 0.068 | 0.005 | 0.004 | 0.083 | 0.166 |
| 5000 | 0.068 | 0.005 | 0.000 | 0.000 | 0.576 |
| 10000 | 0.068 | 0.005 | 0.000 | 0.000 | 0.804 |

  Footnote (p. 44, verbatim): "Results are from 1,000 simulation draws. Confidence bands and
  parallel trends tests are at the 95%-level. Coverage of the uniform bands is uniform
  coverage, i.e., the share of simulation draws in which the bands contained the true
  treatment effects for **all** `t = t* + 1, ..., T`. Coverage of the pointwise bands is the
  share of simulations in which the interval for period `t` contained the true treatment
  effect averaged over `t = t* + 1, ..., T`. The final column contains the share of simulated
  datasets in which the duration and standard parallel trends tests rejects, where the test
  is based on the block bootstrap and uniform confidence bands for the time-varying test
  statistic, as specified in Section 2.3." (The test is specified in Section 3.2 / Algorithm
  2, not Section 2.3.) The paper's `n` is printed bare; the cached script uses `n` **per
  treatment arm** (`n1 = n2 = n`, `montecarlo_revision.m:94-95`), i.e. `2n` observations per
  row.

*Application replication targets (Section 4, pp. 24-28; Appendix D, p. 45; as printed):*
- Austria, reform of 1 August 1989 (data of Lalive et al. 2006). "Following the reform,
  individuals aged 40-49 who had been employed for at least 312 weeks out of the previous ten
  years became eligible for 39 weeks (273 days) of benefits rather than the previous 30 weeks
  (210 days)" (p. 24). Only individuals eligible for the PBD extension and NOT for the
  replacement-rate change are kept.
- Cohorts by `s_i` (unemployment start relative to the reform date): treated `-210 < s_i ≤ 92`
  (5,311 individuals), untreated `-575 < s_i ≤ -273` (5,390) (p. 24, Figure 4.1); windows
  separated by one year (p. 25).
- `t` = days since the spell began; `Y_{i,t}` = "whether that individual had exited
  unemployment at or prior to `t` days after becoming unemployed" (p. 25; inclusive).
- `t* = 209` nominally, shifted to **`t* = 202`** (one week earlier) for anticipation (p. 25).
- Covariate: the calendar day of the year on which the spell began, balanced with (3.8);
  calendar dates with fewer than two untreated individuals dropped (40 of 291) → 4,511
  treated, 5,364 untreated (p. 25).
- `α_t = 0` for the first 152 periods, then `α_t = 1/50` for the subsequent 50 (p. 26);
  expanding the window by 50 days "makes little qualitative or quantitative difference".
- 10,000 block-bootstrap replications; 95% pointwise and uniform bands (Figures 4.3(b),
  4.4(a), p. 27); pre-treatment test with 60% uniform bands (α = 0.40), p-value 0.643
  (p. 28); bands contain zero on days 203-209 (no anticipation evidence, p. 28); the effect is
  "strongly negative immediately following treatment" and attenuates after day 273 (pp. 27-28;
  the p. 27 text attributes the attenuation to the expiry of benefits "for untreated
  individuals ... marked by the solid vertical line", although the solid line is day 273 =
  treated expiry per p. 26 - as printed).
- Figure 4.2 (p. 26): average outcomes and covariate-balanced time-average hazards, with a
  marked jump in the untreated hazard at day 210; Figure 4.3 (p. 27): imputed hazards and
  outcomes; Figure 4.4 (p. 27): effects and the pre-trend statistic over the 50-period window.
- Appendix D (p. 45): the proportional-hazards versions (Figures D.1-D.2) are "almost
  visually indistinguishable" from the common-dynamics figures (p. 26).

**Reference implementation(s):**
- Stata: `durationdid v1.0` (`durationdid.ado:1`), full syntax from the first program copy
  (`:8-22`): `durationdid absorbed_time treatment [if] [in], tstar(#) [covariates(varlist)
  sweights(varname) burnin(1) extrapend(0 = max) spec(cd|ph) breps(1000) level(0.95)
  plevel(0.6) prefix(string) nograph seed(12345)]`. `prefix()` and `nograph` are figure /
  output options with no estimation effect; `extrapend(0)` means "use `floor(max
  absorbed_time)`" (`:41-44`); `level()` and `plevel()` are **confidence levels** as fractions;
  `tstar`, `burnin`, `extrapend` are integer-typed. Input is one row per individual with
  `absorbed_time` = the first period with `Y = 1` (README lines 17-19).
- MATLAB: `durationDiD(absorbed_time, D, X, t_star, burn_in = 1, extrapolation_end =
  floor(max(absorbed_time)), spec = 'common dynamics' | 'proportional hazards',
  bootstrap_replications = 1000, level = 0.95, parallel_level = 0.6, sampling_weights = 1)`
  returning a struct (`durationDiD.m:1-14`); `level` and `parallel_level` are confidence
  levels (`:12-13`).
- R: none. Python: none.
- **Licence:** no LICENSE file → black-box / equation-level reference only; no port.

*Paper vs reference code (commit 202e92ef; line numbers re-derived from the cached files):*
1. **Duplicated, partly unparseable files.** `durationdid.ado` contains two full program
   bodies: copy 1 (`:1-610`, banner `:1`, `program define` `:6`, Mata block `:162-610`) with
   `sweights()`, and copy 2 (`:611-1192`, `program define` `:616`) without it; the two
   `_dd_estimate` bodies are byte-identical. Because an ado file is executed top to bottom
   and copy 2 re-runs `capture program drop` and redefines the program and Mata functions,
   the copy live after loading is most likely copy 2 (the older one, which would reject
   `sweights()`); this reading was not verified by execution. `durationDiD.m` likewise holds
   two copies and is **parse-invalid as cached**: line 187 reads `endfunction out =
   durationDiD(...)` - the `end` closing copy 1's local helper `covariate_balancing`
   (opened `:154`; the top-level function closes at `:151`) is fused to copy 2's header. A
   parse-valid extraction of copy 1 is lines 1-186 **plus a line containing only `end`**
   (equivalently lines 1-187 truncated after its first three characters); any parity exercise
   requires such a documented, non-distributed extraction, or the first Stata program.
2. **Specification-test p-value is code-only.** `durationDiD.m:132`
   `out.p_value = mean(max(boot_delta,[],2) > abs(max(delta./std_delta)))` and
   `durationdid.ado:528` use `abs(max(·))`, not `max|·|` (one-sided against a two-sided
   reference distribution; under-rejects when the largest deviation is negative), a strict
   `>`, and a test set `burn_in..t*` that includes the zero anchor (`δ̂_{t*} ≡ 0`; MATLAB's
   `0/0 = NaN` is skipped by `max`, Mata bumps the zero SD to `1e-100` so the anchor
   contributes 0 to the max). The uniform band `CI_delta` is two-sided, so band and p-value
   can disagree. The paper defines no p-value.
3. **Balancing survivor condition is vacuous.** `durationDiD.m:159-160` and
   `durationdid.ado:218` condition on `absorbed_time >= 1`, which every row with an
   integer duration of at least one (the domain the reference targets) satisfies, so cells
   are balanced over the full group including units absorbed at period 1, not over
   period-1 survivors as (2.25)/(3.8) specify (`at > 1` under the code's `surv = at > t`
   convention). Immaterial in the application (daily durations), material in general.
   (Deferred scope.)
4. **PH `ĉ`.** Paper (3.5) is a weighted least-squares slope with the suspect denominator
   subscript; the code uses the **mean of period-wise ratios** `nanmean(H1./H2)`
   (`durationDiD.m:49`; `durationdid.ado:351-354`). Four candidates, three of them
   implementable - tracked decision 1.
5. **Stata-vs-MATLAB discrepancy at `burnin(1)`, `spec(cd)`.** Mata sets `H1[1] = H2[1] = 0`
   (`ado:339-340`) and keeps the artificial `t = 1` difference of 0 in `mean(vals)`
   (`ado:344-346`; only missing values are filtered), shrinking `ĉ` by `(t*-1)/t*`, and adds a
   live `delta_1 = -anchor` to the test vector (`ado:348`); MATLAB's `nanmean` drops the
   `0/0 = NaN` (`.m:39-47`). Under `ph` the two agree (Mata `0/0 = .`). The paper's (3.2) sums
   over `t = 2..t*`, so MATLAB is paper-faithful; the application's `burnin(152)` is
   unaffected; the Monte Carlo (MATLAB, `burn_in = 1`) skips element 1 explicitly
   (`montecarlo_revision.m:193, :250`).
6. **Critical values.** MATLAB uses the interpolating `quantile` (`.m:122-124, :135-136`);
   Mata takes the `ceil(level·B)`-th order statistic (`ado:510-525`); numerically different.
   The library follows its existing band critical-value convention (a documented choice).
7. **Application scripts.** `UIP_application.do` (`:11-18`): `burn_in 152`, `t_star 202`,
   `cohort_end 273`, `extrap_end 365`, 10,000 reps, cohort cut-offs built from 210
   (`:40-46`), raw-means figure via `absorbed_time <= t` (inclusive). **Both** application
   scripts (`.do:11-13`; `R2.m:7-8`) pass `burn_in 152`, `t_star 202`, and both back-ends
   slice inclusively (`H1[1, burn_in..t_star]`, `durationdid.ado:344, :351`;
   `H1(burn_in:t_star)`, `durationDiD.m:45, :49`), giving a weighted window of 51 horizons
   (`152..202`) against the paper's `α_t = 0` on the first 152 periods and `1/50` on the
   subsequent 50, i.e. `153..202` (p. 26) - an off-by-one shared by every replication path. `UIP_application_
   revision_R2.m`: cohort cut-offs built from `t_star = 202` (`:33, :36` → `-567 / -202`
   instead of the paper's `-575 / -210`), 1,000 reps (`:18`), and its raw-means figure uses the **strict** comparison `Y = periods > absorbed_time` (`:61`, feeding
   `UIP_fig_levels`, the Figure 4.2(a) analogue) against the paper's inclusive "at or prior
   to `t`" definition (p. 25) and the estimator's own `>=` (`.m:31`) - a one-period shift in
   that figure only. **The R2 script cannot run against the cached estimator:** its call
   (`:86`) expects 14 outputs and passes 10 arguments, while both copies of `durationDiD`
   return a single struct (`montecarlo_revision.m:136` uses the struct form).
8. **Monte Carlo script.** Fine grid `T = 100000` over 20 coarse periods with hazards scaled
   `1/T` (paper displays `/(T-1)`); discrete `t_star = floor(50000/5000) = 10` passed to the
   estimator (`:89-90, :136`) vs Table 1's `t* = 11`; the true effect in the first
   post-period is zero by construction (`tau_true(11) = 0`, `:124`); `n` per arm (`:94-95`);
   `level = 0.95` passed in BOTH the band-level and the parallel-level positions (`:12, :136`).
   Two result workbooks: `MCresults_pred.xlsx` (`:353-354`) with bias / MSE on the imputed
   counterfactual mean (`y1preded - y1true`, `:215-216`) and coverage columns but no
   rejection columns; `MCresults.xlsx` (`:357-358`) with `bias_tau` / `MSE_tau`
   (`taued - true_tau`, `:230-231`), the `τ`-based coverage arrays (`:233-234`) and
   `parr_rej` / `parr_rejclass` (`:218-219`). Table 2 prints exactly the second structure and
   its footnote defines coverage on "the true treatment effects", so the `τ`-based workbook is
   Table 2's source (structural reading; the script was not executed). The rejection column
   is band-based (band covers zero at the 95% level, `:193`), not the p-value (recorded at
   `:192`, unused). The classic-DiD comparator's pre-trend statistic centres on `t_star - 1`
   while its bootstrap centres on `t_star` (`:168, :178`).
9. **Never-absorbed units and censoring.** Never-absorbed individuals are coded with an
   `absorbed_time` beyond the extrapolation end and treated as survivors; with the default
   `extrapend = floor(max(absorbed_time))` the sentinel itself becomes the last period, so
   such units are counted as absorbed in the final period (`.m:9, :31`; `ado:43, :305`),
   contradicting the README's description (line 55). **No Kaplan-Meier or censoring handling
   anywhere** despite Section 2.3.
10. **Algorithm 1 vs code.** Algorithm 1 evaluates `τ̂` from `t*` (uniform max over
    `t* ≤ s ≤ T`); the code sets `τ̂_t = 0` for `t ≤ t*` (`.m:57-59`; `ado:361-367`) and takes
    the max over `t*+1..T` (`.m:122-124`; `ado:506-507`). Bootstrap indices are i.i.d. over
    the pooled sample, not stratified by group (`.m:62`; `ado:449`).
11. **Sampling weights, split by reference.** *Stata:* `sw` normalised to mean 1 over the
    whole sample (`ado:406`), multiplied into the balancing weights (`:416`) or used directly
    (`:421`), and the resampled `sw_b` re-normalised and carried through every bootstrap draw
    (`:459-467`). *MATLAB:* with no covariates the raw, un-normalised `sampling_weights` enter
    the point estimate (`.m:22-23`) and every no-covariate bootstrap draw resets
    `weightsb = ones(size(D))` (`.m:75-76`), discarding them; with covariates the balancing
    weights are multiplied by `sampling_weights/mean(sampling_weights)` (`.m:21, :74`).
    Both languages divide weighted survival by `n_k` rather than by the weight sum
    (`.m:34-35`; `ado:309-310`), harmless for balancing weights (mean 1 within group) but not
    for sampling weights normalised over the pooled sample. The paper is silent on sampling
    weights. Balancing weights (both): 1 for treated, density ratio for untreated, cells with
    fewer than two untreated units (unweighted count) zeroed and kept fractions rescaled
    (`.m:171-180`; `ado:241-265`). (Deferred scope.)

**Requirements checklist:**

First estimator (maintainer-approved scope):
- [ ] Two-group `DurationDiD` with common treatment timing; `spec ∈ {"common_dynamics",
      "proportional_hazards"}`; Theorem 1 / Eqs. 3.1-3.5 and (3.7) with `K = 2`
- [ ] Per-period `τ̂_t` for post-treatment grid points; imputed counterfactual means and both
      groups' time-average-hazard curves exposed for plotting (the paper's informal check,
      p. 21)
- [ ] Individual block bootstrap (Algorithm 1): SE = bootstrap SD; symmetric studentised
      pointwise band; uniform band via the max over post horizons
- [ ] Algorithm 2 pre-treatment test over `t_2..t_{j*-1}` with band and the written-out
      p-value; anchor-denominator and per-horizon guards for PH; unavailable-test report
- [ ] `α_t` via equal weights over `t_2..t_{j*}` by default; burn-in label or explicit
      non-negative weight vector (mutually exclusive; renormalised to sum to one)
- [ ] Both input shapes (long binary panel; absorbed-time cross-section via the conversion
      contract), balanced-panel, treatment-constancy, dtype and grid validation
- [ ] Exact-zero survival rules (i)-(v), PH domain `ĉ > 0`, zero-SD and invalid-draw policies,
      negative-implied-hazard warning
- [ ] `n_bootstrap` / `seed` semantics; results on the event-study container with `cband_*`
      fields; joint-NaN inference contract via `safe_inference`
- [ ] Validation: (i) Appendix C DGP recovery vs Table 2 (common dynamics only; `n` per arm;
      95% test level); (ii) a proportional-hazards fixture (a PH DGP with a multiplicative
      gap `c ≠ 1` and known `τ_t`, plus closed-form checks of the `ĉ` candidates (i), (ii) and (iv), including
      zero / near-zero control differences at some horizons); (iii) closed-form CD fixtures;
      (iv) optional reference-code parity on a synthetic no-covariate dataset only via a
      documented extraction of the first MATLAB copy (lines 1-186 + `end`) or the first
      Stata program (equation-level, no port)

Deferred (later PRs; recorded here only):
- [ ] Covariate adjustment: discrete balancing weights (3.8) with the fewer-than-two-untreated
      cell rule, logit propensity weights, sampling weights via `survey_design`, A.4
      semiparametric NLS; weights re-estimated inside every bootstrap draw (p. 23)
- [ ] K-group general linear restrictions (3.6)-(3.7) with constraint-set presets for the
      unrestricted, common-dynamics, proportional-hazards, triple-difference and
      synthetic-control coefficient sets, and the Section 3.3 binding-constraint warning;
      bootstrap bands via Algorithm 1 are source-supported for this mode (generic `τ̂_t`,
      coefficients re-estimated per draw, p. 23), whereas a K-group specification test is
      NOT given by the paper (Algorithm 2 is `k = 1, 2` only) and would be a library
      extension
- [ ] Kaplan-Meier (or other censoring-adjusted) survival input (Section 2.3)
- [ ] Staggered adoption (Appendix A.3) - the paper's own small-group bias caveat, no
      inference
- [ ] Repeated-cross-section inference (Remark 3 gives point estimation only)

---

## Implementation Notes

### Data Structure Requirements

- **Long binary panel (library-native shape):** one row per unit and grid point with the
  unit id, a **numeric** time column (integer or float; datetime64, `pd.Period`, string and
  categorical labels are rejected with a message pointing at conversion to elapsed units,
  unlike library surfaces that accept such labels for ordering only, e.g.
  `diff_diff/estimators.py:743`, `diff_diff/had.py:1353-1360`, `diff_diff/lwdid.py:320-330`),
  a binary absorbing outcome and a 0/1 `treatment` indicator constant within unit. The panel
  must be **balanced over the estimation window** (every unit at every grid point from `t_1`
  through the extrapolation end); genuine right-censoring belongs to the deferred
  Kaplan-Meier input (p. 20), not to silent averaging over whoever is present.
- **Absorbed-time cross-section (the reference's shape) and its conversion contract:**
  duration `d_i` = the first grid time at which the unit is in the absorbing state; numeric;
  domain finite positive reals or `inf` (never absorbed). Evaluation grid `𝒯` = user-supplied
  (uneven grids) or the default integer grid `1..floor(max finite d_i)` (reference
  `durationDiD.m:9`, `:27`). Conversion `Y_{i,t} = 1{d_i ≤ t}` (reference `Y = periods >=
  absorbed_time`, `:31`, so equality means absorbed at `t`, matching the paper's "at or prior
  to `t`", p. 25). `d_i ≤ t_1` → absorbed by the first grid point (all-ones row, counted
  against period-1 survivors); `d_i = inf` → all-zero row; all-`inf` input is rejected.
  **Grid / extrapolation-end precedence:** with an explicit grid, `extrapolation_end` must be
  omitted or equal a grid label (in which case the grid is truncated there); an
  `extrapolation_end` that is not a grid label is rejected together with the explicit grid;
  with the default grid an explicit `extrapolation_end` overrides the default and must lie in
  the data range. **Empty-default-grid guard:** if `floor(max finite d_i) < 1` the default
  grid `1..0` would be empty (the reference performs the same unguarded construction) →
  require an explicit grid or `extrapolation_end`, with a message about the horizon, checked
  before the pre/post-period validation.
- **Repeated cross-sections:** point estimates only (Remark 3, p. 14, is a statement about
  group-period means). Appendix B (p. 38) defines the bootstrap as resampling individuals'
  complete histories, so no repeated-cross-section inference design is source-supported; a
  period-stratified row-resampling scheme would be a documented library extension and is
  deferred.

### Computational Considerations

- Point estimation is `O(n · T)`: one pass to form the survival matrix (or cumulative counts
  of durations), then `O(T)` arithmetic on group means.
- Bootstrap: `B × (n + T)` with `B = 1,000` by default (the paper's application used 10,000);
  parallelisable over draws; deterministic across serial, chunked and parallel execution
  given the seed. The two replicate families (`τ̂*` and `δ̂*`) come from the same draws.
- Numerical: logs of small survival probabilities near the end of the horizon; the exact-zero
  rules above replace any clipping.

### Tuning Parameters

| Parameter | Type | Default | Selection Method |
|-----------|------|---------|-----------------|
| `spec` | `"common_dynamics"` or `"proportional_hazards"` | `"common_dynamics"` | Paper's main text and the reference default; PH avoids implied negative hazards (p. 20); the application reports both (Appendix D) |
| `t_star` | grid label | required | Last pre-treatment grid point (p. 5); shift earlier for anticipation (p. 25) |
| `alpha` | float in (0, 1) | 0.05 | Significance level of the 95% pointwise and uniform bands (paper 95%) |
| `pretrend_alpha` | finite float strictly inside (0, 1); booleans rejected (library `_validate_alpha` shape, `changes_in_changes.py:1160-1164`) | 0.05 | Significance level of the Algorithm 2 band and p-value; the paper's application used α = 0.40 (a 60% band, p. 28); the reference's `plevel` / `parallel_level` (default 0.6) are the complementary confidence levels; the Monte Carlo used 95% (p. 44) |
| `burn_in` | grid label or `None` | `None` (equal weights over `t_2..t_{j*}`) | First grid point with positive weight; the application used the final 50 pre-periods (p. 26); **mutually exclusive with `weights`** (raise if both are given) |
| `weights` | non-negative finite vector of length `j* - 1`, positive total mass | `None` | Explicit `α_t`; **renormalised by its total mass before entering (3.2)/(3.5)** (the paper requires `Σ α_t = 1`, p. 21), with a warning when the supplied sum differs from one beyond tolerance; the default and burn-in constructions already sum to one |
| `pretrend_start` | grid label or `None` | `None` (test set `t_2..t_{j*-1}`) | Optional reference-parity deviation reproducing the code's `burn_in..t*` start; documented as a deviation |
| `extrapolation_end` | grid label or `None` | last grid point / `floor(max finite d_i)` | See the precedence rule above |
| `n_bootstrap` | non-negative int | 1000 | Paper and reference default; **`0` turns inference off**: `τ̂_t`, `ĉ`, hazard curves and imputed means remain available, `se` is NaN and `t_stat` / `p_value` / `conf_int` follow via `safe_inference(att, nan)`, both bands are `None`, the specification test is unavailable (`changes_in_changes.py:1030-1078` precedent); when `> 0`, at least 2 draws (estimator-side floor; `validate_n_bootstrap`, `diff_diff/utils.py:487-504`, accepts 0 and 1) |
| `seed` | validated int or `None` | `None` | Fed to `np.random.default_rng` as in `changes_in_changes.py:1055`; deterministic across execution modes; the reference default is 12345 |
| `stratify` | bool | `False` | Reference resamples the pooled sample; stratified-by-group draws offered as an option |
| covariate / weight source | - | - | Deferred (balancing cells, propensity model, sampling weights) |

### Relation to Existing diff-diff Estimators

The following are facts about the diff-diff library supplied by the caller; they are kept
separate from the paper's claims.

- **No hazard / survival / duration / absorbing-outcome estimator exists** (grep of
  `diff_diff/` and `docs/methodology/REGISTRY.md`; "absorbing" appears only for treatment
  status, e.g. the TROP validator in `diff_diff/trop_local.py`). **Consequence:** this is a
  new estimator family, not an extension of an existing class.
- **`WooldridgeDiD`** (`diff_diff/wooldridge.py`, `method="logit"|"poisson"`) is the library's
  nonlinear-DiD surface and the Remark 4 relative. **Consequence:** it does NOT compute the
  paper's object (no exponential-survival / complementary-log-log link, no group-specific
  linear trend in `R`); the review's Remark 4 transcription documents the connection only.
- **Closest existing bootstrap pattern:** `diff_diff/changes_in_changes.py:735-782` pre-pivots
  to unit-level arrays (after asserting balance at `:734-748`), draws
  `idx = rng.integers(0, n_units, n_units)` and carries each drawn unit's whole history - the
  Appendix B individual block bootstrap - with a row-resampling branch for the
  repeated-cross-section shape; `diff_diff/lwdid.py:4150-4164` resamples cluster / unit keys.
  `stratified_bootstrap_indices` (`diff_diff/bootstrap_utils.py:39-77`) draws treated and
  control strata separately and is therefore only the optional stratified variant.
  **Valid-replicate convention:** joint finite-row mask, `n_valid`,
  `warn_bootstrap_failure_rate` (`bootstrap_utils.py:79`), threshold
  `max(2, _MIN_VALID_REPLICATE_SHARE · n_bootstrap)` (`changes_in_changes.py:241`, `:801-808`).
  **No-bootstrap convention:** point estimate computed before the `n_bootstrap > 0` gate, SE
  NaN in the else branch, then `safe_inference` (`changes_in_changes.py:1030-1078`).
  **Bootstrap p-value convention:** `compute_bootstrap_pvalue` (`bootstrap_utils.py:278-311`),
  inclusive comparisons and a `1/(n_valid + 1)` floor. `compute_percentile_ci`
  (`bootstrap_utils.py:254`) is NOT the paper's interval (the paper's bands are symmetric and
  studentised); the multiplier-bootstrap mixins (`diff_diff/staggered_bootstrap.py`,
  `diff_diff/imputation_bootstrap.py`) are weight-based, not resampling; chunking via
  `diff_diff/bootstrap_chunking.py`. **Consequence:** a new individual-resampling helper is
  needed; it can follow the CiC pattern and share the valid-replicate and p-value helpers.
- **Inference contract:** `safe_inference()` (`diff_diff/utils.py:378-417`) returns
  `(t_stat, p_value, conf_int)` from a point estimate and SE - it never touches `att` or `se` -
  using normal / t critical values; it cannot build the paper's studentised empirical-quantile
  interval, and the paper defines no per-period p-value. **Consequence:** tracked decision 2.
- Reuse: `validate_binary` (`diff_diff/utils.py:52`, with the NaN / empty caveat);
  `validate_n_bootstrap` (`utils.py:487`); `solve_logit` (`diff_diff/linalg.py:3836`) for the
  deferred propensity-weight variant; `BaseEstimator` (`diff_diff/_base.py`) and `BaseResults`
  (`diff_diff/results_base.py`); per-period results on `EventStudyResults`
  (`results_base.py:216`, `time_scale="calendar"`, `cband_lower / cband_upper /
  cband_crit_value`, `post_periods`); `_validate_alpha` shape (`changes_in_changes.py:1160-1164`).
- Conceptual relatives for the deferred Theorem 2 / (3.6)-(3.7) extension: `TripleDifference`
  (`diff_diff/triple_diff.py`; DDD is `β = (·, 1, 1, -1)`), `SyntheticControl`
  (`diff_diff/synthetic_control.py`; simplex weights are `β_1 = 0, β_k ≥ 0, Σ β_k = 1`; its
  Frank-Wolfe simplex solver is reusable), `SyntheticDiD`. None operate on `R` transforms
  today.
- Diagnostics precedent for the Algorithm 2 test: `diff_diff/diagnostics.py`
  (`PlaceboTestResults(Diagnostic)`), `diff_diff/had_pretests.py::joint_pretrends_test`.
- **Column vocabulary** per `docs/v4-design.md` section 8: the time column always means the
  calendar / spell period (rule 1); `unit` names the unit id and the word "group" is reserved
  (rule 3); `treatment` is the 0/1 treated-group indicator (rule 7); `covariates` is the
  covariate list (rule 4); no `_col` suffixes (rule 5); `control_group` with underscored
  values if K-group selection is added (rule 6); `first_treat` only if a staggered lane ever
  exists (rule 2); R/Stata equivalents go in a mapping table rather than aliases (rule 8:
  `absorbed_time treatment, tstar()` → `duration` / `treatment` / `t_star`); results fields
  echoing a parameter share its name (rule 9). Domain names `t_star`, `burn_in`, `spec`,
  `extrapolation_end`, `pretrend_alpha`, `pretrend_start` are legitimate.
- **Reuse vs new, summary:** reuse the CiC unit-block bootstrap pattern, the
  valid-replicate gate, the bootstrap p-value floor convention, `safe_inference`, the
  event-study results container with its uniform-band fields, the base-estimator contracts
  and the alpha / bootstrap-count validators. New: the survival-matrix and time-average-hazard
  transform, the two `ĉ` estimators, the imputation and effect formulas, the studentised
  pointwise and uniform bands, the Algorithm 2 statistic and p-value, the absorbed-time
  conversion contract, and the exact-zero / PH-domain guards.

### Approved initial scope and remaining product decisions

**Approved first-estimator scope (maintainer, 2026-09-05, via the coordinator) - a decision,
not a proposal:** core two-group `DurationDiD` (results `DurationDiDResults`; the library's
CamelCase + `DiD` + `Results` convention as in `ImputationDiD`; the authors' `durationdid`
name goes in the R/Stata mapping only), common treatment timing,
`spec ∈ {"common_dynamics", "proportional_hazards"}`, Theorem 1 / Eqs. 3.1-3.5 and (3.7) with
`K = 2`, per-period `τ̂_t` for post-treatment grid points, imputed counterfactual means and the
time-average-hazard curves exposed for plotting, the individual block bootstrap (Algorithm 1)
with SE, pointwise and uniform bands, the Algorithm 2 pre-treatment test over the paper's
range with the written-out p-value, `α_t` via equal weights by default plus burn-in / explicit
weight-vector options, both input shapes with the conversion contract, never-absorbed
handling, uneven numeric time grids, the validation, minima and edge-case guards above, and
the `n_bootstrap` / `seed` semantics. **Deferred:** covariate adjustment in all forms, K-group
linear restrictions, Kaplan-Meier censoring input, staggered adoption,
repeated-cross-section inference. Deferred items appear in the checklist and Gaps only;
TODO / DEFERRED rows are created by the estimator PR.

**Remaining decisions for the estimator PR's plan, with a recommendation and alternatives
(the first two must be settled before that plan is approved):**
1. **PH `ĉ` formula (blocking):** with `A_t = Δ_{t-1}R̂_{1,t}`, `B_t = Δ_{t-1}R̂_{2,t}`:
   (i) the unscaled weighted-LS slope `Σ α_t A_t B_t / Σ α_t B_t²` (the printed (3.5) with
   the denominator subscript corrected); (ii) the reference code's mean of period-wise
   ratios `Σ α_t (A_t / B_t)` (`durationDiD.m:49`); (iii) the as-printed (3.5) (not
   implementable: equals `1/c` under exact PH); (iv) the **scaled** slope
   `Σ α_t A_t B_t/(t-1)² / Σ α_t B_t²/(t-1)²`, the actual `β_1 = 0`, `K = 2` first-order
   condition of (3.6), which the paper's p. 21 sentence implies (3.5) should be; (i) and (iv)
   coincide only up to `(t-1)^{-2}` re-weighting of the horizons. Candidates (i), (ii) and
   (iv) are consistent for `c` under exact PH - the as-printed (iii) converges to `1/c`,
   which is why it is not implementable - and the three implementable forms differ in finite
   samples only through horizon weighting. Recommendation:
   (iv) as the default, because it is the form the paper's own general procedure (3.6)
   produces and is what a future K-group mode reduces to, with (i) and (ii) as options ((ii)
   for parity with the authors' software); whichever form is selected owes a REGISTRY
   `- **Note:**` covering both the denominator-subscript correction and the scale / weighting
   choice; in all cases validate the PH domain `ĉ > 0`.
2. **Inference construction (blocking):** recommendation - keep the library's joint
   convention: per-period `att / se / t_stat / p_value / conf_int` from
   `safe_inference(τ̂_t, σ̂_boot,t)` (normal reference); store the paper's studentised
   pointwise band and the uniform band as separate fields (`cband_*` for the uniform band,
   a named pointwise-band pair alongside), with a REGISTRY `- **Note:**` that the paper
   defines only bootstrap bands and no p-value. Alternative: put the studentised pointwise
   band in `conf_int` as a labelled deviation from the joint-inference convention.
3. Default `spec` (`common_dynamics`, with a warning when the imputed `Δ_{t-1}R̂^{(0)}_{1,t}`
   decreases in `t`).
4. `pretrend_alpha` default (0.05, matching the library's `alpha` convention and the paper's
   Monte Carlo; the application's α = 0.40 is documented as an application choice).
5. Bootstrap defaults (1,000 draws; unstratified as in the reference, with `stratify` as an
   option).
6. Headline scalar `att` (the paper defines no aggregate; recommend the mean of
   post-treatment `τ̂_t` with a bootstrap SE and a labelled estimand; alternative: no
   aggregate).
7. REGISTRY category "Advanced Estimators"; results fields `c_hat`, `spec`, `t_star`, hazard
   curves, imputed means, `spec_test_p_value`, `spec_test_crit_value`, `delta` band.

---

## Gaps and Uncertainties

1. **No analytic variance and no formal asymptotic theorem.** Section 3.3 (pp. 23-24)
   asserts GMM / sequential-GMM validity and omits a formal statement "following the example
   of Wooldridge (2023)". Only the bootstrap is available.
2. **No p-value is defined.** Sections 3.2 and Appendix B specify band tests only; the
   p-value the application reports (0.643, p. 28) comes from the authors' code (fidelity
   item 2). The library p-value above is an extension.
3. **Two printed forms of the pre-treatment statistic.** Section 3.2 (p. 23) prints the `t*`
   reference terms as `Δ_{t-1}R_{k,t*}/(t-1)`; Algorithm 2 (p. 40) prints
   `Δ_{t*-1}R̂_{k,t*}/(t*-1)`; the code follows Algorithm 2 (`durationdid.ado:348/:355`,
   `durationDiD.m:47/:51`).
4. **Algorithm 2 labels and ranges.** Step 7 calls the sup-t quantile "the pointwise level
   1 - α critical value"; step 8 writes the band around `τ̂_t` instead of `δ̂_t`; the text
   rejects over `2 ≤ t ≤ t*` while the algorithm uses `2 ≤ t ≤ t* - 1`; step 4 writes
   `Y_{t,j_b(i)}` and `G_{t,j_b(i)}` (p. 40).
5. **Algorithm 1 index ranges.** Step 1 prints "`t = t*, , ..., T`"; steps 6-7 define `σ̂_t`
   and pointwise critical values for `t = t*+1..T` while the uniform max runs over
   `t* ≤ s ≤ T`, so `σ̂_{t*}` is used but never defined (p. 39); the code takes the max over
   `t*+1..T` (fidelity item 10).
6. **(2.9)-(2.10) index start** (p. 11): stated for `1 < t ≤ T` but undefined at `t = 2`
   under (2.8); (2.10) carries no printed range.
7. **`α_t` positivity vs zero weights** (p. 21 "positive" with `Σ = 1`, then "`α = 0` for
   early observations"; p. 26): implementable domain = non-negative weights with positive
   total mass, renormalised to one. The application's "first 152 periods" + 50 = 202 = `t*`
   counts periods from `t = 1`, consistent with the `t = 1` lower limit printed in (3.6)
   rather than the `t = 2` of (3.2).
8. **(3.5) as printed**: denominator `Σ α_t Δ_{t-1}R̂²_{1,t}` (equals `1/c` under exact PH; a
   least-squares slope would carry `ΔR̂²_{2,t}`) and `R_{1,1}` without a hat (p. 21); the
   code uses a further estimator (mean of ratios). **The p. 21 claim that (3.2) and (3.5)
   "are both special cases of the procedure below" (3.6) holds for (3.2) but not for (3.5)
   as printed**: the `β_1 = 0` case of (3.6) is the scaled slope with `(t-1)^{-2}` horizon
   weights, which differs from both the printed and the subscript-corrected unscaled
   (3.5). Whichever form the estimator PR selects owes a REGISTRY `- **Note:**` covering the
   denominator-subscript correction and the scale / weighting choice. Tracked as decision 1.
9. **(3.6) lower limit `t = 1`** where the summand is `0/0` (p. 21); the prose says
   "regressing `ΔR̂_{1,t}`" while the display regresses the scaled long difference.
10. **Theorem 2 continuation** (p. 16): "`ΔR^{(0)}_{1,t}` is identified by `R^{(0)}_{1,t} = …`"
    names a first difference and defines a level; (2.12) drops the `(0)` superscripts that
    (2.11) carries (p. 12); the Theorem 2 proof (p. 47) states "`R^{(0)}_{1,t} = R_{1,t}`"
    where the following display uses `R_{1,1}`.
11. **Appendix A.3 baseline** (p. 34): the PH identification display starts from `R_{k,t}`
    while the CD display and both plug-in estimators start from `R_{k,1}`; `R_{k,1}` is the
    (A.6)-consistent baseline.
12. **Appendix A.4 slips**: `(1 - t)β_1` after (A.12) where `(t - 1)β_1` follows from (A.12)
    and Theorem A.1 (p. 37); **(A.13) prints its second expectation without the `| G_i = 1`
    conditioning that the p. 50 proof derives and the p. 38 `1{G_i = 1}/n_1` analogue
    implements** (p. 37); the empirical ATT analogue weights by `(1 - Y_{it})` where (A.13)
    weights by `(1 - Y_{i,1})`, and sums over an index `n` (p. 38); Theorem 4's full-support
    condition reads "conditional on `Y_{i,t} = 0`" (p. 36); the appendix theorem is numbered
    "Theorem 4" after "Theorem A.1".
13. **Proof slips**: `exp(tc)` for `exp((t-1)c)` in the Proposition 1 proof (p. 48; cancels in
    the final step); the p. 50 display divides by `-φ(0)` where (E.6) requires `+φ(0)`.
14. **Cross-reference and label slips**: Table 2's footnote cites "Section 2.3" for the test
    (it is Section 3.2 / Algorithm 2; p. 44); Appendix C prose cites "Figure A.1-A.4" and
    "Figure 1.a" for figures captioned C.1-C.4 and Figure 1.1 (pp. 41-43); Appendix A.2 says
    "Theorem 2 below" (p. 32); the p. 13 figure explanation measures the ATT against "the
    solid red curve" (Group 2) where p. 9 uses the solid and dashed blue lines; the p. 27
    text attributes the attenuation after day 273 to the expiry of benefits "for untreated
    individuals" although the solid line marks treated expiry (p. 26); Appendix D's last
    sentence is a fragment (p. 45); the synthetic-control restriction adds "sum to 1" only on
    p. 10; Appendix A.4.1 cites "the assumptions and results in Section 2.2.2" (p. 37), a
    section that does not exist (Section 2.2 has only 2.2.1); Algorithm 1 step 1 says
    "evaluate the estimator `τ̂_t` as in Section 3.1" (p. 39), where Section 3.1 is the
    covariate-adjustment subsection and the estimator itself is in the Section 3 preamble.
15. **DGP timing**: the factual hazard uses `1{t ≥ t*}` and the prose says the intervention
    "occurs at time `t*`" (p. 41) against the p. 5 convention "strictly after `t*`"; the
    switching-probability prose says "between `t - 1` and `t`" while the display is the
    `t → t + 1` transition (p. 41); Table 2 assesses coverage over `t*+1..T` only (p. 44).
    The cached script discretises `t*` to 10 (Table 1 prints 11) and its first post-period
    true effect is zero (fidelity item 8).
16. **Table 2 provenance** rests on a structural reading of the script's two workbooks (the
    `τ`-based `MCresults.xlsx` matches the printed columns); the script was not executed.
    The rejection column is band-based at the 95% level, not p-value-based.
17. **No `α_t` selection recipe** beyond "closer to the intervention" or "minimize the
    asymptotic variance" (p. 21); no `α_k` recipe under staggered adoption (p. 34).
18. **Censoring** is only sketched (Kaplan-Meier-adjusted survival as input, p. 20); no
    recipe for combining it with the bootstrap; the reference code has none.
19. **Proportional hazards degeneracy**: identification needs one nonzero pre-treatment
    control difference (footnote 5, p. 13); the paper does not discuss zero denominators in
    the per-horizon PH statistic or an in-sample `ĉ ≤ 0`, which is outside the model's
    strictly-positive-hazard domain (p. 10). Common dynamics may imply negative hazards
    (p. 20, main text).
20. **Staggered adoption (A.3)** has no inference procedure, no `K > 2` or staggered version
    of Algorithm 2, and a stated small-group bias caveat (p. 35). **Semiparametric A.4** has
    no inference recipe and requires a known link, full conditional support and no intercept
    (footnote 10).
21. **Repeated cross-sections**: point identification only (Remark 3); the block bootstrap
    presumes individual histories (p. 38).
22. **Binding inequality constraints** in `𝓑` invalidate the bootstrap (Section 3.3); no
    alternative is implemented in the paper.
23. **Code conventions that are not paper statements**: the fewer-than-two-untreated cell
    rule and its rescaling, sampling weights, the `burnin`-anchored test window, the
    one-sided p-value, the pooled (unstratified) resampling, `τ̂_t = 0` hard-coded for
    `t ≤ t*`, the never-absorbed sentinel convention. The live Stata copy (fidelity item 1)
    was not verified by execution.
24. **Licence absent** on the reference repository; the R2 application script targets an
    earlier 14-output function that the cached files do not provide.
25. **Publication status and v1 facts** rest on cached secondary web sources (the authors'
    pages and the arXiv v1 listing, all accessed 2026-09-05); re-check on publication.
26. **Extraction-coverage note.** Five extraction passes covered pp. 1-16, 13-28, 32-41,
    40-50 and the six code files; the bibliography (pp. 28-32) was not extracted beyond the
    eight entries on p. 28 and the two on p. 32 (pp. 29-31 unread by design). No page failed
    to render; the only legibility caveats are small-print panel titles in Figure C.4 and the
    band legend of Figure D.2(b) (which appears to read "60% Uniform CIs"; uncertain). No
    contradictions between extractors were found beyond the as-printed items above.
