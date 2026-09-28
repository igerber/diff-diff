# Paper Review: Panel Data and Experimental Design

**Authors:** Fiona Burlig, Louis Preonas, Matt Woerman
**Citation:** Burlig, F., Preonas, L., & Woerman, M. (2020). Panel Data and Experimental Design. *Journal of Development Economics*, 144, 102458.
**DOI:** https://doi.org/10.1016/j.jdeveco.2020.102458
**Source reviewed:** NBER Working Paper No. 26250 (September 2019), the openly-available pre-publication version, at `https://www.nber.org/system/files/working_papers/w26250/w26250.pdf` (also Becker Friedman Institute WP 2019-113 and Haas WP-277). The published *J. Dev. Econ.* 144 (2020) version is paywalled; **equation and page numbers below refer to the NBER WP w26250** and may be renumbered in the journal version. PDF was reviewed externally and is **not** committed to the repository (the `/papers/` working directory is gitignored). Accompanying reference software: the Stata package **`pcpanel`** (from SSC; https://econpapers.repec.org/software/bocbocode/s458286.htm).
**Review date:** 2026-05-31

---

## Methodology Registry Entry

**Status: proposed/confirming source text for the `## PowerAnalysis` REGISTRY entry; this file is a
non-authoritative source audit.** The current `docs/methodology/REGISTRY.md` `## PowerAnalysis` block
remains the sole authoritative methodology contract. This review establishes what Burlig, Preonas &
Woerman (2020) *actually* derive so that a follow-up audit PR (PR-B) can reconcile the REGISTRY block
and `diff_diff/power.py` against it. The registry-candidate text ends just before `## Implementation
Notes`; everything below that boundary is audit notes and is **not** normative.

Scope note: Burlig et al. (2020) is the primary source for **the variance of the panel
difference-in-differences (DD) estimator under (non-constant) serial correlation** — the "panel" path
of the library's power calculations. It is *not* the source for the MDE-multiplier framing itself (see
[bloom-1995-review.md](bloom-1995-review.md)) nor for survey design effects (Kish 1965, at REGISTRY
Kish DEFF note). Its lineage: it generalizes **Frison & Pocock (1992)** and **McKenzie (2012)** to arbitrary
serial correlation.

## PowerAnalysis

**Primary source:** [Burlig, F., Preonas, L., & Woerman, M. (2020). Panel Data and Experimental Design. *Journal of Development Economics*, 144, 102458.](https://doi.org/10.1016/j.jdeveco.2020.102458)

**Key implementation requirements:**

*MDE definition (Eq. 1, WP p.3):* uses the **t distribution** (finite-sample), two-sided test:

```
MDE = ( t^d_{1-kappa} + t^d_{alpha/2} ) * sqrt( Var(tau_hat | X) )          (Eq. 1, p.3)
```

where `t^d_{α/2}` is the t critical value with `d` degrees of freedom for Type-I error `α`,
`t^d_{1−κ}` the value for power `κ`, and `d` depends on the dimensions of `X` and the estimator
(footnote 7). For a **one-sided** test, `t^d_{α/2}` is replaced by `t^d_α` (footnote 7, p.3).

*Model (Assumptions 1-5, WP pp.4-5):*
- DGP (Assumption 1): `Y_it = β + τ·D_it + υ_i + δ_t + ω_it`, with `υ_i ~ iid N(0,σ²_υ)` (unit shock),
  `δ_t ~ iid N(0,σ²_δ)` (time shock), `ω_it ~ N(0,σ²_ω)` idiosyncratic (**not** necessarily iid — this
  is where serial correlation lives), `τ` homogeneous. `J` units, proportion `P` treated, `m`
  pre-treatment and `r` post-treatment periods (Assumption 3, balanced panel).
- Estimator: OLS with unit and time fixed effects, `τ̂ = (D̈'D̈)⁻¹D̈'Ÿ` (p.5).
- Symmetric within-unit covariance averages (Assumption 5, p.5) — the three "ψ" terms:
  ```
  ψ^B = avg pre-treatment within-unit Cov(ω_it, ω_is)        (Before)
  ψ^A = avg post-treatment within-unit Cov(ω_it, ω_is)       (After)
  ψ^X = avg cross-period within-unit Cov(ω_it, ω_is)         (pre x post)
  ```
  assumed equal across treated and control groups (`ψ^B = ψ^B_T = ψ^B_C`, etc.).

*The serial-correlation-robust (SCR) variance — the paper's core result (Eq. 2, WP pp.5-6):*

```
MDE = ( t^J_{1-kappa} + t^J_{alpha/2} ) * sqrt( Var(tau_hat | X) )

                       1          [  m+r              m-1            r-1                      ]
Var(tau_hat|X) = ------------- *  [  --- * sigma^2_w + --- * psi^B + --- * psi^A  -  2*psi^X ]   (Eq. 2)
                  P(1-P) J        [   mr               m              r                       ]
```

Edge rule (footnote 11, p.6): if `m = 1` then `ψ^B` is undefined and its term is multiplied by 0;
likewise if `r = 1` for `ψ^A`.

*McKenzie (2012) special case (Eq. 3, WP p.6)* — the SCR formula with `ψ^B = ψ^A = ψ^X = 0`
(iid idiosyncratic errors after fixed effects):

```
                                                  ( sigma^2_w     m+r )
MDE = ( t^J_{1-kappa} + t^J_{alpha/2} ) * sqrt(    ---------- *  ---  )                      (Eq. 3)
                                                  ( P(1-P) J      mr  )
```

*Sign / direction of serial correlation (Eq. 4, WP p.8):* `ψ^B` and `ψ^A` enter the MDE **positively**
(serial correlation in pre or post periods *erodes* the benefit of extra waves), while `ψ^X` enters
**negatively** (cross-period correlation makes DD differences easier to detect — it *reduces* the MDE).
Net, serial correlation *increases* the MDE **iff**:

```
( (m-1)/m ) * psi^B + ( (r-1)/r ) * psi^A  >  2 * psi^X                                      (Eq. 4)
```

Otherwise it *decreases* the MDE. With positive serial correlation in longer panels, adding time
periods can actually **increase** the MDE (p.7, footnote/WP §2.1.2) — the opposite of the iid
intuition.

*Lemma 1 (WP p.6):* the SCR variance is an unbiased estimator of `E[Var(τ̂|X)]` even under arbitrary
**within-period cross-sectional** (cross-unit, same-time) correlation, under unit-level randomization.

*Allocation:* the `P(1−P)` factor implies power is maximized at `P = 1/2` (50/50), as in the
cross-sectional case (consistent with [bloom-1995-review.md](bloom-1995-review.md)).

*Paper-derived requirements checklist:*
- [ ] Panel-DD MDE uses the **t** distribution (Eq. 1); one- and two-sided supported.
- [ ] Panel variance separates `m` (pre) and `r` (post) periods (Eq. 2), not a single period count.
- [ ] Variance carries three covariance terms `ψ^B, ψ^A, ψ^X`, with `ψ^X` entering **negatively**.
- [ ] iid / constant-serial-correlation reduces to the McKenzie form (Eq. 3).
- [ ] `m = 1` or `r = 1` zero out `ψ^B` / `ψ^A` respectively (footnote 11).
- [ ] Power maximized at `P = 1/2`.

---

## Key Results

| Result | Statement | Implementation relevance |
|--------|-----------|--------------------------|
| Eq. 1 (p.3) | `MDE = (t^d_{1−κ}+t^d_{α/2})·√Var` | t-based MDE multiplier (cf. Bloom's z) |
| Eq. 2 (p.5-6) | SCR DD variance with `ψ^B, ψ^A, ψ^X` over `m, r` | the panel-DD variance contract |
| Eq. 3 (p.6) | McKenzie special case (`ψ=0`) | iid baseline; `(m+r)/(mr)` period factor |
| Eq. 4 (p.8) | when serial correlation raises the MDE | direction (raise/lower) cannot be one-signed |
| Lemma 1 (p.6) | unbiased under within-period cross-sectional correlation | robustness of the SCR variance |

---

## Implementation Notes (audit notes — NOT registry-candidate)

These observations map Burlig et al. (2020) to `diff_diff/power.py` and the current REGISTRY block.
They are flagged here for **PR-B** to reconcile (fix-vs-document); this review changes no code/REGISTRY.

- **D4 — the code's panel factor is NOT Burlig's SCR formula; the Burlig attribution is an
  overclaim.** The code's panel branch (`_compute_variance`, `power.py`) computes
  `Var = σ²(1/n_T+1/n_C)·(1+(T−1)ρ)/T` with `T = n_pre+n_post` and a single ICC `ρ`. Rewriting
  `1/n_T+1/n_C = 1/(P(1−P)J)`, this is `σ²/(P(1−P)J)·(1+(T−1)ρ)/T` — the variance of a **mean of `T`
  equicorrelated observations** (Moulton/equicorrelated design effect), monotonically **increasing**
  in `ρ`. It (a) collapses `m` and `r` into a single `T`, contradicting Eq. 2's separate pre/post
  treatment; (b) has no analogue of `ψ^X` and so **cannot represent that cross-period correlation
  lowers the MDE** (Eq. 4 / p.8) — Burlig's central DD insight; (c) is neither Eq. 2 (SCR) nor Eq. 3
  (McKenzie). PR-B candidates: document the panel path as a deliberate equicorrelated/Moulton
  simplification and **re-attribute** (it is not Burlig's SCR formula — cite Burlig as the SCR method
  the library does *not* yet implement), or implement Eq. 2/Eq. 3. Decision returns to the user.
- **D1 — t vs z, with both papers in view.** Burlig's Eq. 1 uses the **t** distribution (`t^d`/`t^J`);
  Bloom (1995) uses the **normal**; the code (`_get_critical_values`, `power.py`) uses the
  **normal** (`stats.norm.ppf`). So the code is a large-`J` normal approximation to Burlig's t-form.
  The REGISTRY's `t_{α/2}+t_{1−κ}` notation (the REGISTRY PowerAnalysis block) matches Burlig; the code deviates to `z`. PR-B
  candidate: document the normal-approximation deviation (valid for large `J`), or switch the panel
  path to `t` with appropriate `d`.
- **`basic_did` boundary.** The code's 2×2 branch `2σ²(1/n_T+1/n_C) = 2σ²/(P(1−P)J)` equals McKenzie
  Eq. 3 at `m = r = 1` (`(m+r)/(mr) = 2`). But the panel branch evaluated at `T=2` gives
  `0.5σ²/(P(1−P)J)` — a 4× discontinuity with `basic_did`. They are used in disjoint regimes
  (`T ≤ 2` vs `T > 2`), so no runtime bug, but it confirms the two branches rest on different sampling
  models (per-cell counts vs per-unit equicorrelated mean). Flag for PR-B.
- **R/Stata parity reference for the panel path is `pcpanel`, not `pwr`.** Burlig's reference
  implementation is the Stata `pcpanel` package (SSC). Any PR-B panel-path parity fixture should target
  `pcpanel` (which implements both the McKenzie and SCR formulas), not `pwr` (cross-sectional, t-based).
- **Allocation `P(1−P)`** in Eq. 2/3 matches the code's `f(1−f)` (`_compute_required_n`, `power.py`)
  and Bloom's `T(1−T)` — consistent across all three sources.

## Gaps and Uncertainties

1. **WP vs published equation numbers.** Equation/page numbers above are from NBER WP w26250 (Sept
   2019). The published *J. Dev. Econ.* 144 (2020) version (paywalled) may renumber; PR-B should
   confirm against the published article if it transcribes equation numbers into REGISTRY.
2. **ANCOVA estimator out of library scope.** Burlig also derives SCR power for the ANCOVA estimator
   (WP §2.2.2, pp.18+), which is often preferred in short panels (Frison & Pocock 1992; McKenzie 2012).
   The library has no ANCOVA estimator; this is noted but out of scope for the current `power.py`.
3. **Lineage citations not yet in REGISTRY.** If PR-B documents the code's panel path as the
   constant-correlation / equicorrelated case, the appropriate primary citations for *that* case are
   Frison & Pocock (1992) and/or McKenzie (2012) (Eq. 3), which Burlig generalizes — these are not
   currently in `docs/references.rst`. Whether to add them is a PR-B decision (sourced from this paper's
   explicit lineage, not domain knowledge).
4. **The simulation path** (`simulate_*`, whose docstring cites this paper at `power.py`) estimates
   power by Monte Carlo (fit estimator + count rejections) and does **not** implement Eq. 2; Burlig uses
   Monte Carlo only to *validate* the analytical SCR formula (WP §2.1.2). Whether the simulation path
   warrants its own citation is flagged for PR-B (it is outside what an analytical reading of this paper
   can adjudicate).
