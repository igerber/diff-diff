# rddensity 3.0 source notes (R source consulted - not a paper review)

**What this is:** implementation-level notes on the R `rddensity` 3.0 package
source, the parity target of `diff_diff/rddensity.py` (`RDDensityTest`).
Unlike the files in `docs/methodology/papers/`, there is no PDF behind this
document: the source of truth is the CRAN source tarball
(`https://cran.r-project.org/src/contrib/rddensity_3.0.tar.gz`, sha256
`a9c45ab0f6b86ead4d91084db16513d4156b7f59b0472510b63deb5dee6f305d`), files
`rddensity/R/{rddensity,rdbwdensity,rddensity_fun}.R`. These notes record the
software-level behaviors that the CJM 2020 paper (and its supplemental
appendix, reviewed in
`docs/methodology/papers/cattaneo-jansson-ma-2020-review.md`) does not
specify, discovered by reading and executing the pinned source.

## The rank-based EDF

R's regressand is `Y <- (0:(N-1))/(N-1)` computed on the FULL sorted sample -
a rank-based EDF mapping the smallest observation to 0 and the largest to 1.
The CJM 2020 paper displays `F_tilde(x) = (1/n) sum_i 1[x_i <= x]`; the two
differ by up to `1/N` pointwise. The port implements R's form (REGISTRY
Note). Consequence for the joint-vs-separate identity: a side-specific fit
(the side's own rank EDF) equals the joint fit's side times
`(N-1)/(n_side - 1)`, not the paper's `n/n_side`.

## Mass-point adjustment (`massPoints=TRUE`, the default)

When repeated values exist: the EDF is evaluated on the UNIQUE sorted values
and replicated back to the full sample via the last-occurrence index
(`rddensityUnique(...)$indexLast` + `freq`); the jackknife's leave-one-out
projection additionally replicates its reverse-cumsum rows via the
FIRST-occurrence index (`indexFirst`). Point estimates and standard errors
both change. R warns at `summary()` time; the port warns at fit time
(no-silent-failures convention) and exposes the RD-family string surface
`masspoints="adjust"/"check"/"off"` mapping R's boolean.

## The window-restricted jackknife

R builds the leave-one-out projection matrix `L` from the WINDOW rows only:
`L[, j] <- (cumsum(c(0, XpW[Nh:1, j]))/(N-1))[Nh:1]` - row `i` sums the
weighted design rows strictly after `i` within the estimation window. This
DIFFERS from the CJM supplemental appendix's Section 5.2 literal `U_hat`
double sum, in which out-of-window observations remain eligible as the
non-localized member of each pair (see the review's computational note). The
port ships R's windowed construction (parity target); the difference is
locked by a methodology test.

## The variance surface

R 3.0 implements exactly two `vce` modes: `"jackknife"` (the default; the
windowed leave-one-out construction above) and `"plugin"` (DGP-free moment
matrices with the estimated densities plugged in). The SA Theorem-2
"automatic" triple-sum estimator is NOT in the package. The restricted-model
plugin builds the minus-side second-moment matrix as the UNCORRECTED
reflection `Gm <- Psi %*% G %*% Psi` - the SA Lemma 13 variance form; the
corrected `Gamma_minus` (SA proof 7.17, with two rank-one correction terms)
is never used by any estimation path (`Gminusgenerate` is dead code in the
3.0 namespace, referenced only by the unused `h_opt_density_res`).

## Degenerate-design behavior

`rddensity_fV` solves with `solve(crossprod(XpW, Xp), tol = 0)` - the rank
check is DISABLED. On a singular Gram R throws internally and returns an
all-NA frame; on rank-deficient-but-solvable designs (e.g. a side of only
repeated values: rank 5/8, condition ~5e16) R silently returns numerically
meaningless finite estimates, and finite NEGATIVE side densities can be
reported with finite standard errors. The port replaces both silent modes
with fail-loud guards (unique-support precondition per fV call; solve
failures/non-finite outputs raise) and a warning on finite negative
densities (REGISTRY Deviations).

## Bandwidth selector mechanics

- Normal-reference preliminary bandwidths: `bn` (for `F^(p+1)`, order `p+2`)
  and `cn` (densities, order `p`) from the hard-coded per-`p` constant
  vectors `Cb`/`Cc` (p = 1..7) and Hermite-polynomial normal plug-ins
  evaluated at the standardized sample mean, scaled by the sample SD.
- Preliminary regularization: two gates, `if (nLocalMin > 0)` and
  `if (nUniqueMin > 0)`; EACH gate floors BOTH `bn` (hard-coded count
  `20+p+3`) and `cn` (count `20+p+1`) - the gates differ only in the sample
  quantiled (full `X` vs unique values). User-supplied floor VALUES enter
  only the final-stage regularization; a zero disables the gate.
- The h-table: `h = ((1/(2p)) * V / B^2 / N)^(1/(2p+1))` per row
  (left/right/diff/sum). `rddensity_fV` NaNs negative variances before the
  selector sees them, so the selector's `hn[i,2] < 0` branch is dead code;
  an NaN bandwidth is zeroed by the `is.na` branch, while a bias-squared row
  of EXACTLY zero (restricted sum row under exact per-side cancellation)
  yields `h = Inf`, which `is.na` misses - it survives to the range clamp
  when `regularize=TRUE` (clamping to the observed range) and would flow
  into estimation under `regularize=FALSE`. The port replicates the NaN->0
  and Inf-passes-to-clamp behavior exactly and adds a fail-loud
  finite-and-positive guard on the SELECTED bandwidths.
- `bwselect="comb"`: per-side `median(each, diff, sum)` (unrestricted) or
  `min(diff, sum)` (restricted). Manual-h fits record R's `bwselectl`
  value, whose literal spellings are `"estimated"` and the package typo
  `"mannual"` (the port's `bandwidth_method` maps them to
  `"estimated"`/`"manual"`).

## Out of the port's scope (documented seams)

The binomial windows test (R's `bino=` block; methodology from Cattaneo,
Frandsen & Titiunik 2015 / Cattaneo, Frandsen & Vazquez-Bare 2017, not the
reviewed CJM 2020 paper) and `rdplotdensity` (requires an lpdensity port).
Tracked in `DEFERRED.md`.
