# Paper Review: Minimum Detectable Effects — A Simple Way to Report the Statistical Power of Experimental Designs

**Authors:** Howard S. Bloom
**Citation:** Bloom, H. S. (1995). Minimum Detectable Effects: A Simple Way to Report the Statistical Power of Experimental Designs. *Evaluation Review*, 19(5), 547-556.
**DOI:** https://doi.org/10.1177/0193841X9501900504
**Source reviewed:** *Evaluation Review* 19(5), 547-556 (10 pages). PDF was reviewed externally and is **not** committed to the repository (the `/papers/` working directory is gitignored). Reproduce by downloading the published article via the DOI above, or the open scan used here at `https://bpb-us-e2.wpmucdn.com/sites.uci.edu/dist/1/1159/files/2021/03/Bloom-MDES-Eval-Rev-1995-Bloom.pdf`. Page numbers below refer to the journal pagination (547-556).
**Review date:** 2026-05-31

---

## Methodology Registry Entry

**Status: proposed/confirming source text for the `## PowerAnalysis` REGISTRY entry; this file is a
non-authoritative source audit.** The current `docs/methodology/REGISTRY.md` `## PowerAnalysis` block
remains the sole authoritative methodology contract. This review establishes what Bloom (1995)
*actually* states so that a follow-up audit PR (PR-B) can reconcile the REGISTRY equation block and
`diff_diff/power.py` against it. The registry-candidate text ends just before `## Implementation
Notes`; everything below that boundary is audit notes and is **not** normative.

Scope note: Bloom (1995) is the primary source for **the minimum-detectable-effect (MDE)
multiplier framing** and for **the cross-sectional (two-group) impact-estimator standard error**. It
is *not* a difference-in-differences paper, and it **explicitly excludes** clustering / multi-site
design effects (Note 1, p.555) — those rest on other sources (serial correlation: Burlig, Preonas &
Woerman 2020, see [burlig-preonas-woerman-2020-review.md](burlig-preonas-woerman-2020-review.md);
survey design effect: Kish 1965, already in REGISTRY).

## PowerAnalysis

**Primary source:** [Bloom, H. S. (1995). Minimum Detectable Effects. *Evaluation Review*, 19(5), 547-556.](https://doi.org/10.1177/0193841X9501900504)

**Key implementation requirements:**

*Definitions (p.547):*
- The **minimum detectable effect (MDE)** is "the smallest effect that, if true, has an X% chance of
  producing an impact estimate that is statistically significant at the Y level," where X = statistical
  power and Y = significance level (p.547).
- The MDE is measured **in the original units of the impact** (e.g. dollars), explicitly *not*
  standardized like Cohen's (1977) effect size (p.547-548).

*MDE multiplier — derived from the NORMAL distribution (p.548-549):*

Bloom constructs the multiplier from "two normal (bell-shaped) sampling distributions" (p.548). For a
one-sided test at the .05 level with 80% power he derives:

```
critical value (one-sided .05):  1.65 standard errors above zero   = z_{0.95}
power shift (80%):               0.84 standard errors above crit.   = z_{0.80}
MDE = (1.65 + 0.84) * SE = 2.49 * SE                                (p.549)
```

The general rule (p.549-550): "For any significance level, power value, and one- or two-sided
hypothesis test, the minimum detectable effect can be computed as a multiple of the standard error of
the impact estimate." In standard-normal-quantile form (all multipliers below reproduce Bloom's
stated values exactly using `z` quantiles, confirming the normal — not t — basis):

```
MDE = M * SE(impact)
M_one_sided = z_{1-alpha}   + z_{power}
M_two_sided = z_{1-alpha/2} + z_{power}
```

*Table 1 multipliers explicitly stated in the text* (p.549-551), one-sided test at the .05 level:

| Power | Multiplier M | Check (z_{0.95} + z_{power}) |
|-------|--------------|------------------------------|
| 90%   | 2.93         | 1.645 + 1.282 = 2.927        |
| 80%   | 2.49         | 1.645 + 0.842 = 2.487        |
| 70%   | 2.17         | 1.645 + 0.524 = 2.169        |

One-sided .10 at 80% power: M = 2.12 (p.552) = z_{0.90} + z_{0.80} = 1.282 + 0.842 = 2.124.
Table 1 has a top panel (one-sided) and a bottom panel (two-sided); the columns are significance
levels and the rows are power (p.550).

*Standard error of the impact estimator (p.551):*

Bloom gives Equation (1) for a **continuous** outcome from a regression-adjusted treatment/control
difference of means (simple random sample), and Equation (2) for a **binary** outcome. The typeset
equations are image-only in the available scan; their algebraic form is fixed unambiguously by Bloom's
explicit textual description (p.551) and Note 8 (p.556). With:
- `sigma` = standard deviation of the continuous outcome
- `Pi`    = proportion with outcome value 1 (binary case)
- `T`     = fraction of the sample randomly assigned to treatment
- `n`     = total study-sample size
- `R^2`   = explanatory power of the impact regression

```
Eq (1)  continuous:  SE_c = sigma * sqrt( (1 - R^2) / ( n * T * (1 - T) ) )
Eq (2)  binary:      SE_b = sqrt( Pi*(1 - Pi) * (1 - R^2) / ( n * T * (1 - T) ) )
```

Bloom states Equation (1) "increases as heterogeneity sigma increases, decreases as R^2 increases,
decreases as n increases" (p.551) and (Note 8, p.556) the MDE "increases in inverse proportion to the
square root of T(1-T)" — pinning the `(1-R^2)` numerator and the `n*T*(1-T)` denominator. Equation (2)
"differs from Equation (1) only in that the population variance is expressed as Pi(1-Pi) … instead of
sigma^2" (p.551). Writing `n_T = nT`, `n_C = n(1-T)` gives the equivalent two-group form
`Var = sigma^2 (1-R^2) (1/n_T + 1/n_C)`.

*Treatment/control allocation (p.553-554, Table 2):*
- Statistical power is **maximized at a 50/50** treatment/control mix (p.553).
- MDE rises *slowly* away from 50/50: 60/40 → 1.02x, 70/30 → 1.09x the 50/50 MDE (p.554, Note 9);
  by Note 8 the MDE scales as `1/sqrt(T(1-T))`, which is symmetric in `T ↔ 1-T` (Note 7).

*One-sided vs two-sided (p.554-555):*
- Bloom argues program evaluation should use a **one-sided** test (decision-oriented), which has a
  smaller MDE / higher power than two-sided (p.554-555), but provides multipliers for **both** (Table 1).

*Paper-derived requirements checklist:*
- [ ] MDE computed as `M * SE` with `M` from the **normal** distribution (one- and two-sided).
- [ ] Cross-sectional SE supports the `(1-R^2)` covariate-adjustment factor and the `T(1-T)` allocation factor.
- [ ] Binary-outcome variance available as `Pi(1-Pi)` in place of `sigma^2`.
- [ ] Power maximized at 50/50; MDE robust to moderate allocation imbalance.
- [ ] One-sided and two-sided multipliers both supported.
- [ ] Clustering / multi-site design effects are **out of scope** for Bloom's Eq (1)-(2) (Note 1).

---

## Implementation Notes (audit notes — NOT registry-candidate)

These observations map Bloom (1995) to `diff_diff/power.py` and the current REGISTRY block. They are
flagged here for **PR-B** to reconcile (fix-vs-document); this review does not change code or REGISTRY.

- **D1 (t vs z) resolves in the code's favor.** Bloom's multiplier is built entirely from the normal
  distribution (p.548-549). `PowerAnalysis._get_critical_values` (`power.py`) uses
  `stats.norm.ppf` — **faithful to Bloom**. The REGISTRY block writes the multiplier as
  `(t_{alpha/2} + t_{1-kappa})` (the REGISTRY PowerAnalysis block); per the primary source this should be `z`
  (normal), not `t`. PR-B candidate: correct the REGISTRY notation to `z`, or document the
  normal-approximation explicitly.
- **R-parity reference implication.** Because the analytical path uses the normal approximation (per
  Bloom), `pwr::pwr.t.test()` (noncentral-t) is **not** the right parity reference for it; a
  normal-based reference (`pwr::pwr.norm.test` / `pwr.2p2n.test`, or a hand-derived closed form) is.
  This bears on the deferred PR-B R-parity fixture choice.
- **Bloom's SE is cross-sectional, not DiD.** Eq (1) `Var = sigma^2(1-R^2)(1/n_T+1/n_C)` is a
  single-measurement two-group estimator. The code's `basic_did` branch
  (`_compute_variance`, `power.py`) uses `sigma^2(1/n_T+1/n_T+1/n_C+1/n_C) = 2 sigma^2(1/n_T+1/n_C)`
  — the **DiD analog** (two independent time points, factor of 2), with **no `R^2` term**. So Bloom
  underpins the MDE *multiplier* and the cross-sectional SE; the DiD variance itself is a separate
  (DiD-specific) quantity. PR-B candidate: reconcile the REGISTRY SE formula (which currently shows
  `sigma*sqrt(1/n_T+1/n_C)*sqrt(1+rho(m-1))/sqrt(1-R^2)`, the REGISTRY PowerAnalysis block) against the code — note the
  `R^2` term appears **inverted** there relative to Bloom (Bloom multiplies by `sqrt(1-R^2)`; the
  REGISTRY divides by it). The code's `basic_did` variance uses the group-count form
  `2*sigma^2*(1/n_T+1/n_C)` — allocation still enters, implicitly via the `n_T`/`n_C` counts here and
  explicitly as `f(1-f)` in `_compute_required_n` (`power.py`) — rather than Bloom's
  total-`n` x `T(1-T)` parameterization, and it omits the `R^2` factor entirely.
- **Allocation factor is faithful.** Bloom's `T(1-T)` (Note 8) appears in the code's required-N
  formula as `f(1-f)` (`_compute_required_n`, `power.py`); the REGISTRY sample-size formula
  `n = 2(...)^2 sigma^2 / MDE^2` (the REGISTRY PowerAnalysis block) **omits** it. PR-B candidate: add the allocation factor to
  the REGISTRY formula.
- **Binary outcomes.** Bloom's Eq (2) (`Pi(1-Pi)`) is supported in the library only by the user
  passing `sigma = sqrt(Pi(1-Pi))`; there is no dedicated binary path. Reasonable simplification;
  worth a one-line note for PR-B.
- **Out-of-scope for Bloom.** `deff` (Kish survey design effect) and the panel `rho` serial-correlation
  factor are **not** Bloom — Note 1 (p.555) explicitly says Eq (1)-(2) "do not account for the design
  effect … in multisite experiments" and points to other sources. See the Burlig review for the panel
  variance; Kish (1965) is already cited at the REGISTRY Kish DEFF note for `deff`.

## Gaps and Uncertainties

1. **Typeset Eq (1)/(2) and Table 1 are image-only in the available scan.** The algebraic forms above
   are reconstructed from Bloom's explicit prose (p.551) + Notes 8/9 (p.556) and verified to reproduce
   every numeric multiplier Bloom states (2.93/2.49/2.17/2.12). If PR-B needs the exact typeset
   coefficients of the full Table 1 grid (all power × significance × one/two-sided cells), obtain a
   text-layer copy of the article; the values are nonetheless fully determined by `M = z_{1-alpha(/2)} + z_{power}`.
2. **Numeric SE verification not possible from this scan.** Bloom's worked Examples 1-3 (p.552-553)
   report SEs ($560, 2.8 scale points, 0.040) but the example *parameters* are image-only, so the SE
   formula could not be re-derived numerically from the paper. The form is standard and textually
   pinned; this is a transcription limitation, not a methodological doubt.
3. **DiD applicability.** Bloom does not treat panel/DiD designs; the library's DiD variance and its
   serial-correlation handling must be validated against the Burlig review, not this one.
