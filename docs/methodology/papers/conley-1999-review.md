# Paper Review: GMM Estimation with Cross Sectional Dependence

**Authors:** Timothy G. Conley
**Citation:** Conley, T. G. (1999). GMM Estimation with Cross Sectional Dependence. *Journal of Econometrics*, 92(1), 1-45. DOI: 10.1016/S0304-4076(98)00084-0
**PDF reviewed:** papers/1-s2.0-S0304407698000840-main.pdf
**Review date:** 2026-05-09

---

## Methodology Registry Entry

*Formatted to match docs/methodology/REGISTRY.md structure. Heading levels and labels align with existing entries - copy the `## ConleySpatialHAC` section into the appropriate category in the registry.*

## ConleySpatialHAC

**Primary source:** Conley, T. G. (1999). GMM Estimation with Cross Sectional Dependence. *Journal of Econometrics*, 92(1), 1-45.

**Scope:** Cross-sectional spatial heteroskedasticity-and-autocorrelation-consistent (HAC) covariance matrix estimator for GMM/OLS when observations are realizations of a random field on a Euclidean space (typically R^2). Each observation `i` is associated with a location `s_i` and the dependence between `X_{s_i}, X_{s_j}` is allowed to be a (decreasing) function of the "economic distance" `d_{ij}`. The estimator is the spatial analog of Newey-West (1987) / Andrews (1991): a kernel-weighted sum of pairwise outer products, truncated beyond a user-supplied bandwidth. Conley distinguishes two settings: (i) Section 3, locations on an integer lattice with exactly observed distances; (ii) Section 4, real-valued locations with bounded measurement error in distances. For diff-diff Phase 1, only the OLS-just-identified specialization (Section 5 empirical example) is needed.

**Why this paper is foundational for diff-diff:** every spatial-HAC SE used in modern applied DiD work traces back to this paper's three core ingredients — (a) a kernel-weighted pairwise meat with bandwidth `h`, (b) consistency under a polynomially-decaying mixing condition with bandwidth growth rate `o(n^{1/3})`, and (c) tolerance to bounded measurement error in distances (so applied users can use approximate haversine on lat/lon). Stata's `acreg` (Colella et al. 2019), `conleyreg` (Düsterhöft 2021), and the Hsiang (2010) MATLAB code all implement specializations of Equation 3.13 / 4.2.

**Key implementation requirements:**

*Assumption checks / warnings:*
- Random-field framing: each observation `i` is associated with a position `s_i` in R^2 (or general R^l), and `X_{s_i}` is a stationary random field with mixing coefficients that decay in inter-point distance (Section 2 page 5; Section 3.1.1 page 6).
- Sample region `Λ_τ` must be a sequence of finite closed convex regions that grow uniformly in at least two non-opposing directions (Assumption A1, page 8). Footnote 5 (page 4) flags the importance of two-direction growth: with growth along one direction only the data could be handled like a time series.
- Mixing condition on the random field: `α_{k,l}(n)` is the standard `σ`-algebra mixing coefficient with index sets of size `≤ k`, `≤ l` and minimum Euclidean distance `≥ n` (Equation 3.1, page 7). The required tail decay is `α_{1,∞}(m)^{δ/(2+δ)} = o(m^{-2})` and `Σ_m m·α_{k,l}(m) < ∞` for `k+l ≤ 4` (conditions B1-B3 page 9). Practical interpretation: the autocovariance function must die off polynomially in distance.
- Moment condition: for some `δ > 0`, `E[ ‖g(X_s; β)‖^{2+δ} ] < ∞` (B3, page 9). For OLS this becomes `E[ ‖x ε‖^{2+δ} ] < ∞`.
- Sampling/directing process `W_s` (Section 3.1.2 page 6): `W_s` is independent of the underlying random field `X_s`, stationary, mixing, with `E W_s = λ` and zero otherwise. This represents the irregular spacing of agents on the lattice (`H ∩ Λ_τ`); it can also accommodate cluster sampling so long as it is independent of `X`. Section 6 (page 22) flags this independence as a substantive restriction; "endogenous locations" violate it.
- Real-valued locations (Section 4): require a minimum economic distance `d_0` between distinct agents (Assumption D4 "hard core" point process, page 14) and bounded mixing measurement error in distances (Assumptions E1-E2, page 15).
- Warn (do NOT fit silently) when a user supplies `conley_cutoff_km = 0` (or `h = 0`): the estimator collapses to White HC0 (with no diagonal-only adjustment for the `j=k=0` double-counting subtraction, see Equation 3.13).
- Warn when the user-supplied cutoff exceeds roughly half the diameter of the sample region. Conley does not state a hard rule but the empirical example (Section 5, page 21) sweeps from 60-90 (truncated window) and 75-225 (Bartlett window) on a sample of 95 countries with distance metric centered around 50-150 between economically close pairs (Table 1, page 20).

*Variance estimator (Equation 3.13 page 12, as implemented for OLS):*

For the GMM moment process `Y_{m,n}(β) := g(X_{s_i}; β)` indexed by lattice coordinates `(m,n)`:

    Ĉ_τ = (1/T_τ) · Σ_{j=0}^{L_M} Σ_{k=0}^{L_N} Σ_{m=j+1}^{M} Σ_{n=k+1}^{N} K_{MN}(j,k)
          · [ Y_{m,n}(b_τ) Y_{m-j, n-k}(b_τ)' + Y_{m-j, n-k}(b_τ) Y_{m,n}(b_τ)' ]
          - (1/T_τ) · Σ_{m=1}^{M} Σ_{n=1}^{N} Y_{m,n}(b_τ) Y_{m,n}(b_τ)'                    (3.13)

The second term subtracts the doubled `j = k = 0` contribution introduced by the symmetrized first sum (page 11 above Equation 3.13). For OLS, `Y_{m,n}(b) := x_i (y_i - x_i' b) = x_i ε̂_i`. The "meat" matrix `Ĉ_τ` is then sandwiched:

    Var̂(β̂) = (X'X)^{-1} · Ĉ_τ · (X'X)^{-1}     (just-identified GMM, Section 5 framing)

This is the spatial analog of the time-series HAC estimator (Newey-West 1987; Andrews 1991): a weighted sum of pairwise outer products with weights `K_{MN}(j,k)` that vanish when the lattice index gap exceeds `L_M` (in the M-direction) or `L_N` (in the N-direction).

For real-valued locations with possible measurement error in distances (Equation 4.2 page 18):

    C̃_τ = (1/T_τ) · Σ_{j=0}^{L_M} Σ_{k=0}^{L_N} Σ_{m̃=j+1}^{M̃} Σ_{ñ=k+1}^{Ñ} K(j/L_M, k/L_N)
          · [ Y_{m̃,ñ}(b_τ) Y_{m̃-j, ñ-k}(b_τ)' + Y_{m̃-j, ñ-k}(b_τ) Y_{m̃,ñ}(b_τ)' ]
          - (1/T_τ) · Σ_{m̃=1}^{M̃} Σ_{ñ=1}^{Ñ} Y_{m̃,ñ}(b_τ) Y_{m̃,ñ}(b_τ)'                  (4.2)

where `K(·,·)` is a bounded continuous function on `[-1,1] × [-1,1]` with absolutely summable Fourier coefficients and `K(0,0) = 1`. The plane is partitioned into squares of side `d̄ < d_0` and observations are relabeled by their square coordinates `(m̃, ñ)`. (For diff-diff this is the relevant form: locations are real-valued lat/lon, distances are computed pairwise rather than via lattice indexing.)

In a pairwise pseudo-distance form that is more familiar to applied users (and that diff-diff will implement directly):

    C̃_τ = Σ_{i, j} K(d_{ij} / h) · X_i ε̂_i ε̂_j X_j'

with `K(0) = 1` (so the `i = j` term contributes `X_i ε̂_i² X_i'`, which equals the White HC0 contribution). This is the form Conley sketches in Section 4.3 (Equation 4.2 plus the "pairwise products at a given distance" remark on page 19) and is the form used by every downstream Stata/R/MATLAB implementation. The exact mapping: with a 2-D Bartlett window `K(j/L_M, k/L_N) = (1 - |j|/L_M)(1 - |k|/L_N)` (Equation 3.14), the kernel is separable in M and N coordinates; in pairwise form it becomes a function of the L_∞ distance scaled by `(L_M, L_N)`. For a single isotropic cutoff `h` and Euclidean (or great-circle) distance, the standard practitioner specialization is `K(d_{ij}/h)`.

where:
- `d_{ij}` = distance between units `i` and `j` (Conley: economic distance, possibly distorted by bounded mixing measurement error)
- `K(·)` = kernel function
- `h` = bandwidth (cutoff). Conley uses `(L_M, L_N)` for direction-specific bandwidths
- `X_i ε̂_i` = OLS score for observation `i`
- `T_τ` = number of observations in sample region `Λ_τ` (Conley uses `T_τ` for sample size, NOT `n`)

*Kernel functions (Section 3.3.1 pages 11-12, Section 4.3 page 18):*

The class of kernels Conley considers must satisfy (conditions C1, page 12; restated p.18 for the real-valued case):
- Uniformly bounded: `|K_{MN}(j,k)| ≤ const`
- Convergence to one at the origin: `K_{MN}(j,k) → 1` as `τ → ∞` for each fixed `(j,k)`. In the scaled form `K(j/L_M, k/L_N)`, this is `K(0,0) = 1`.
- Bandwidth growth: `L_M = o(M^{1/3})` and `L_N = o(N^{1/3})` (condition C1, page 12). This is the spatial analog of Andrews' `1/4` rate restriction in the time series case.
- For PSD point estimates (Section 3.3.1, p.12): the weights must correspond to a non-negative spectral window, equivalently the Fourier transform of `K(·,·)` must be non-negative. Bartlett, Parzen and similar windows from time-series HAC literature (see Priestley 1981) inherit this property.

Specific kernels named in the paper:

- **Truncated (uniform) window** (page 11, cited as White 1984's truncated estimator):

      K_{MN}(j,k) = 1{|j| < L_M, |k| < L_N}

  Easy to construct from imprecise distance information ("constant over the distances within a category"), but its spectral window (the Fourier transform) is "negative in some regions" (footnote 11, page 11) so it is NOT generally PSD.

- **Bartlett window (Newey-West, 2-D product form)** (Equation 3.14, page 12):

      K_{MN}(j,k) = (1 - |j|/L_M)(1 - |k|/L_N) · 1{|j| < L_M, |k| < L_N}

  Its Fourier transform is non-negative (page 12), so `Ĉ_τ` is PSD with this weighting. The empirical example (Section 5, page 21) uses this kernel with truncation points 75-225 alongside a truncated window with cutoffs 60-90.

- **General Bartlett-class** (sketched, not formalized): "many others (see, e.g., Priestley 1981) for other examples" (page 18).

- **Pairwise (1-D) Bartlett (the form diff-diff will implement)** is not explicitly written in the paper. The 2-D `(1 - |j|/L_M)(1 - |k|/L_N)` is the only PSD kernel formula in the paper. The 1-D form `K(u) = max(0, 1 - |u|)` is the standard practitioner specialization (Hsiang 2010; Colella et al. 2019) and reduces to Conley's 2-D Bartlett along the M-axis when `L_N → ∞` (or vice versa). See "Gaps" section.

Bandwidth selection (Section 3.3, Section 4.3):
- **Rate condition (consistency)**: `L_M = o(M^{1/3})`, `L_N = o(N^{1/3})` (C1, page 12). Bandwidth must grow with sample size but slower than the cube root.
- **No plug-in / cross-validation rule given.** Conley does NOT supply a data-driven bandwidth selector. The empirical example (Section 5 page 21) reports estimates over a coarse grid and notes "the qualitative results discussed below are robust to changes in this cutoff value and/or window specification." This is the standard sensitivity check approach that downstream packages (acreg, conleyreg) inherit.
- **Practical guidance from the empirical example (page 21)**: "Some idea of the relative magnitude of this truncation value of 75 is afforded by Table 1." Table 1 (page 20) shows the economic distances: USA-Mexico = 32, UK-France = 53, USA-Japan = 119, USA-Algeria = 141, USA-Pakistan = 218. So a cutoff of 75 includes near neighbors (USA-Mexico, UK-France) but excludes distant pairs. The implementation should expose `conley_cutoff_km` directly (no auto-selection) and document Conley's robustness-grid recommendation.

*Edge cases (compiled across paper):*
- **`d_{ij} = 0` for `i ≠ j` (multiple observations at same coordinates)**: page 19 - "If measurements of economic distances do not locate agents in distinct locations, this strategy to get PSD estimates cannot always be implemented. However, `C̃_τ` will still remain consistent as long as the measurement errors satisfy conditions E1 and E2." Page 18 (eq 4.2 footnote): when there are multiple observations with the same index (large `d̄` or non-distinct distance measurements), the bracketed term `[Y_{m̃,ñ}(b_τ)_1 Y_{m̃-j, ñ-k}(b_τ)_1' + Y_{m̃-j, ñ-k}(b_τ)_1 Y_{m̃,ñ}(b_τ)_1']` gets expanded to all the cross products at that distance. Practical handling: the implementation should sum all `X_i ε̂_i ε̂_j X_j'` cross terms at zero distance the same way as nonzero distance.
- **No spatial dependence (`α_{k,l}(m) = 0` for all `m > 0`)**: Conley does NOT explicitly note this case, but inspection of Equation 3.13 with `K(j,k) → 1{j=k=0}` (i.e., cutoff `h = 0` with no ties) reduces to `(1/T_τ) Σ_m Σ_n Y_{m,n} Y_{m,n}' = (1/n) Σ_i x_i ε̂_i² x_i'`, which is White (1980) HC0 (the meat matrix). So at `h = 0` and no spatial ties Conley reduces to HC0 (page 19 names HC0 as "Eicker, 1967; Huber, 1967; White, 1980").
- **Ties in distance (multiple cluster-like structure)**: when `K(d_{ij}/h)` is the uniform indicator and `d_{ij}` is `0` for units in the same group and `> h` otherwise, Conley reduces to cluster-robust (CR0) - specifically to the score-outer-product cluster sum without the small-sample `G/(G-1)` finite-sample correction. This reduction is not explicit in the paper but is implied by Equation 3.13 with `K = 1{d_{ij} = 0}`.
- **Imperfect distance measurement** (Section 4): bounded measurement error in `d_{ij}` is OK as long as E1-E2 hold; the estimator stays consistent (Proposition 5, page 19). This justifies practitioner use of approximate haversine distance from lat/lon.
- **PSD failure with truncated kernel**: "This estimator will not always be PSD, unfortunately, since the spectral window corresponding to the step function space domain window (its Fourier transform) will be negative in some regions" (footnote 11, page 11). Implementation should warn (or fall back to Bartlett) if the user requests `conley_kernel="uniform"` and the resulting matrix has negative eigenvalues.
- **Endogenous locations**: Section 6 (page 22) flags `W_s ⊥ X_s` (independence of sampling/directing process from random field) as a substantive restriction. "Allowing for endogenous locations is likely to require an explicit model of how locations and variables in the moment conditions are jointly determined." DiD spillover applications where treatment assignment is correlated with location violate this; the variance estimator may be inconsistent.
- **Sample region "very nearly a line"** (page 28 in proof): the count of "far apart" terms in the variance bound is `MN - 4 · min(L_M, L_N) - 1`, which is the worst case. Two-direction growth of `Λ_τ` (A1) is required to control these boundary terms.

*GMM moment condition and estimator (Section 3.2 pages 7-8):*

Population moment condition (Equation 3.2):

    E[ g(X_s; β) ] = 0

where `β` is `k × 1` and `g: R^l × B → R^v` with `v ≥ k`. The `v - k` overidentifying restrictions are exploited via the GMM objective (Equation 3.3):

    J(b)_τ = [ (1/T_τ) Σ_{i=1}^{T_τ} g(X_{s_i}; b) ]' Ω_τ [ (1/T_τ) Σ_{i=1}^{T_τ} g(X_{s_i}; b) ]    (3.3)

minimized over `b ∈ B`. For OLS (Section 5 page 19), `g(x_i, y_i; β) = x_i (y_i - x_i' β)`, `v = k`, `Ω_τ = I` (just-identified), and the minimizer is the OLS estimator.

*Consistency conditions (Proposition 1 page 8):*

Under conditions A1-A3, `b_τ → β` in probability as `τ → ∞`:
- **A1** (sample region): `Λ_τ` grows uniformly in two non-opposing directions as `τ → ∞`. (Page 8.)
- **A2** (weighting matrix): `Ω_τ → Ω` in probability where `Ω` is positive-definite. (Page 8.)
- **A3** (regularity): `X_s` and `W_s` are mixing; `g(·; b)` is Borel measurable for all `b ∈ B`; `g(X; ·)` is continuous on `B` for all `X ∈ R^l`; first-moment continuous on `B`. (Page 8 with footnote 7 defining first-moment continuity.)

*Asymptotic distribution (Proposition 2 page 10, Equation 3.11):*

    sqrt(T_τ) · (b_τ - β) ⇒ N(0, D_0' Λ^{-1} V D_0)    as τ → ∞                  (3.11)

where `D_0` and `V` are the GMM asymptotic-variance ingredients:

    D_0 = { E[Dg(X_s; β)]' Ω E[Dg(X_s; β)] }^{-1} E[Dg(X_s; β)]' Ω
    V = Σ_{s ∈ Z^2} cov(Y_0(β), Y_s(β))

(page 10 Equation 3.12: "the asymptotic covariance matrix of the moment conditions that needs to be estimated is `C ≡ λ^{-1} V`.") `λ = E W_s` is the limiting fraction of lattice points sampled, accounting for the irregular spacing.

For OLS just-identified (Section 5 page 19): `β̂ = (X'X)^{-1} X'y`, score = `x_i ε_i`, `Dg = -x_i x_i'`, so

    Var̂(β̂) = (X'X / T_τ)^{-1} · λ^{-1} V̂ · (X'X / T_τ)^{-1} / T_τ
            = (X'X)^{-1} · Ĉ_τ · (X'X)^{-1}

with `Ĉ_τ` from Equation 3.13. (The `λ` factor is absorbed by the lattice indexing; in the pairwise form `Σ_{i,j} K(d_{ij}/h) · ...` it does not appear.)

*Consistency of the covariance matrix estimator (Proposition 3 page 12):*

Under conditions A1-A3, B1-B5, and C1-C3:

    Ĉ_τ → C    in probability as τ → ∞

where the relevant conditions are:
- **B1**: `Σ_{m=1}^∞ m · α_{k,l}(m) < ∞` for `k + l ≤ 4` (page 9).
- **B2**: `α_{1,∞}(m) = o(m^{-2})` (page 9).
- **B3**: For some `δ > 0`, `E[‖g(X_s; β)‖^{2+δ}] < ∞` and `Σ_m m · α_{1,1}(m)^{δ/(2+δ)} < ∞` (page 9).
- **B4**: `Dg(X_s; b)` is Borel measurable for all `b ∈ B`, continuous on `B` for all `X ∈ R^l`, first-moment continuous; `E Dg(X_s; β)` exists and is full rank (page 10, condition stated atop).
- **B5**: `V = Σ_{s ∈ Z^2} cov(Y_0(β), Y_s(β))` is non-singular (page 10).
- **C1**: `K_{MN}(j,k)` are uniformly bounded; `K_{MN}(j,k) → 1` as `τ → ∞` (M, N → ∞); `L_M = o(M^{1/3})` and `L_N = o(N^{1/3})` (page 12).
- **C2**: For some `δ > 0`, `E[‖g(X_s; β)‖^{4+δ}] < ∞` and `α_{∞,∞}(m)^{δ/(2+δ)} = o(m^{-4})` for both `X_s` and `W_s` (page 12). Note this is the **strengthened** moment / mixing condition relative to B2-B3.
- **C3**: `E sup_B ‖Y_{m,n}(b)‖² < ∞` and `E sup_B ‖(∂/∂b)[Y_{m,n}(b)]‖² < ∞` (page 12). This bounds the score and its gradient uniformly in `b`, so plug-in `b_τ → β` does not destroy consistency.

For real-valued locations with measurement error (Proposition 5 page 19): under A1-A4 (where A4 is uniform growth in two non-opposing directions for the constructed lattice region `Λ_τ*`), B1-B5, C1-C3, D1-D6 (point process assumptions, page 14), E1-E2 (bounded mixing measurement error, page 15), `C̃_τ → C` in probability.

The point-process assumptions D1-D6 (page 14) are:
- **D1**: `X(s)` and the point process `Φ` are independent.
- **D2**: `Φ` is simple (zero or one points at each location w.p.1).
- **D3**: `0 < EΦ(A) < ∞` for all `A` with finite Lebesgue measure.
- **D4**: `Φ` is "hard-core" — no points within `d_0` of each other w.p.1.
- **D5**: `Φ` is stationary.
- **D6**: `Φ` is mixing, with a rate that links to the random field's mixing via a relabeled lattice process `ψ_p = Φ(A_p)`.

*Empirical example numerics (Section 5, Table 2 page 21):*

Cross-country growth regression on 95 countries (Barro 1991 specification): `growth(1960-85) ~ GDP60 + SEC60 + PRIM60 + g^c/Y + REV + ASSASS + PPI60DEV + Africa + LatinAmerica`. Economic distance = transportation cost between countries (Conley & Ligon 1995). Standard errors computed three ways:

| Variable    | Point estimate | IID S.E. | HET S.E. | Spatial S.E. |
|-------------|----------------|----------|----------|--------------|
| Constant    | 0.0333         | 0.0063   | 0.0070   | 0.0053       |
| GDP60       | -0.0066        | 0.0010   | 0.0009   | 0.0008       |
| SEC60       | 0.0124         | 0.0106   | 0.0077   | **0.0019**\*\* |
| PRIM60      | 0.0274         | 0.0060   | 0.0060   | 0.0065       |
| g^c/Y       | -0.0959        | 0.0260   | 0.0269   | 0.0359       |
| REV         | -0.0208        | 0.0085   | 0.0081   | 0.0072       |
| ASSASS      | -0.0024        | 0.0029   | 0.0018   | **0.0009**\*\* |
| PPI60DEV    | -0.0139        | 0.0051   | 0.0048   | **0.0064**\* |
| Africa      | -0.0107        | 0.0038   | 0.0041   | 0.0046       |
| Latin America | -0.0137      | 0.0033   | 0.0032   | 0.0028       |

Spatial SEs use a truncated window with cutoff 75 (countries less than 75 units apart are nonzero, more distant are zero). Sweep range: 60-90 (truncated) and 75-225 (Bartlett). Six of nine spatial SEs are SMALLER than IID/HET counterparts. Conley emphasizes (page 21): "spatial dependence does not imply that standard errors will rise. ... asymptotic variances may be smaller with spatially dependent data, just as asymptotic variances can be lower for dependent time series averages than independent series averages." This is a non-obvious finding for practitioners and worth surfacing in diff-diff documentation/tutorials.

*Algorithm (pairwise form for OLS, the diff-diff implementation target):*
1. Fit OLS: `β̂ = (X'X)^{-1} X'y`, residuals `ε̂ = y - Xβ̂`.
2. For each pair `(i, j)` of observations:
   a. Compute `d_{ij}` via the configured metric (haversine for lat/lon in km; euclidean for projected coords; user callable allowed).
   b. Compute kernel weight `w_{ij} = K(d_{ij} / h)` where `h = conley_cutoff_km`.
   c. Skip pair if `w_{ij} == 0` (sparse fast path: pairs with `d_{ij} ≥ h` for kernels with compact support).
3. Form the meat: `S = Σ_{i,j} w_{ij} · (x_i ε̂_i)(x_j ε̂_j)'`. Note: with the standard Bartlett or uniform kernel and `K(0) = 1`, the `i = j` term contributes `x_i ε̂_i² x_i'` which is the HC0 diagonal.
4. Form the bread: `B = (X'X)^{-1}`.
5. Sandwich: `Var̂(β̂) = B · S · B`.
6. Optional: project to nearest PSD if eigendecomposition reveals negative eigenvalues (only for non-PSD kernels like uniform/truncated).

For the lattice form (Equation 3.13), the weighting array `K_{MN}(j,k)` is indexed by lattice gaps, not pairwise distances. diff-diff's pairwise form is mathematically equivalent when the kernel is separable but is the more natural API for irregular real-valued locations.

**Reference implementation(s):**
- **Stata `acreg`** (Colella, Lalive, Bauer & Thoenig 2019, working paper "Inference with Arbitrary Clustering"): implements the pairwise form with Bartlett or uniform kernel. Cited as the modern Stata-canonical reference for diff-diff Phase 1.
- **Stata `conleyreg`** (Düsterhöft 2021, SSC archive): MATA-coded; supports haversine distance directly.
- **MATLAB `ols_spatial_HAC.m`** (Hsiang 2010, supplementary code for Hsiang 2010 PNAS): the original applied-econ reference implementation used by climate / development economists.
- **R `conleyreg`** (Düsterhöft 2021, CRAN port).
- **No reference R implementation of Conley appears in CRAN as of the paper's 1999 publication.** Modern R alternatives include `lfe::felm` (with `cmethod="cgm2"`, which is cluster-robust not Conley) and standalone Conley scripts circulated by Hsiang and Fetzer.

The paper itself does NOT distribute code. Conley's Section 5 empirical example is reported numerically in Table 2 (page 21) but the underlying replication routine is not in the paper.

**Requirements checklist:**
- [ ] Coordinates supplied as two columns (lat, lon) or `(x, y)` projected.
- [ ] Distance metric configured (haversine for lat/lon; euclidean for projected; callable for custom).
- [ ] Cutoff `conley_cutoff_km > 0` (or unitless `conley_cutoff` for euclidean). Document that `h = 0` reduces to HC0.
- [ ] Kernel choice `conley_kernel ∈ {"bartlett", "uniform"}`. Conley's explicit PSD Bartlett (Eq 3.14) is the 2-D separable lattice product window; the radial 1-D pairwise Bartlett that diff-diff and R `conleyreg` implement is a practitioner specialization that is **not** formally PSD-guaranteed. Uniform is also not PSD in general. Apply the negative-eigenvalue warning to **both** kernels.
- [ ] Score outer products `x_i ε̂_i` computed identically to HC0 path.
- [ ] Robustness sweep: document that practitioners should report estimates at multiple cutoffs (Conley Section 5 standard).
- [ ] If the resulting Conley meat / variance has any materially negative eigenvalues (under either Bartlett or uniform), warn the user (the implementation does this for both kernels).

---

## Implementation Notes

### Data Structure Requirements
- New required columns when `vcov_method="conley"`: two coordinate columns named via `conley_coords=("lat","lon")`. Both must be finite floats. Reject NaN/Inf (no silent dropping, per `feedback_no_silent_failures`).
- Pre-fit checks: confirm coordinate columns exist and are numeric; confirm `conley_cutoff_km > 0`; confirm `conley_kernel ∈ {"bartlett", "uniform"}` (or callable signature); validate metric callable returns nonnegative scalar for two coordinate vectors.
- For two-way space x time (Phase 2 scope), additional time-key column needed; not in Phase 1.

### Computational Considerations
- **Dense distance matrix is O(n²)** in both compute and memory. Conley's discussion (Section 4.3) frames this in terms of lattice-square indexing rather than pairwise distance, but the practitioner-canonical pairwise form realizes the full O(n²) cost.
- **Memory**: For `n = 10000` units and `float64`, the dense distance matrix is `8 · 10^8` bytes = 800 MB; the pairwise outer-product accumulator is `O(n² · k)` floats. Phase 1 should warn at `n > 5000` and refuse (or require explicit override) at `n > 50000`.
- **Sparse fast path (Phase 2)**: With Bartlett or uniform kernel, pairs at `d_{ij} ≥ h` contribute zero. A k-d tree (`scipy.spatial.cKDTree.query_ball_tree` with radius `h`) returns only neighbor pairs, reducing complexity to `O(n · k_avg)` where `k_avg` is the average number of neighbors within `h`. This matches the `acreg` "neighbors-only" inner loop.
- **Parallelization**: pairwise sums are embarrassingly parallel by `i`. Rust backend (Phase 2+) can process row-blocks in parallel.
- **PSD projection**: For non-PSD kernels (uniform), eigendecomposition + clamping negative eigenvalues to zero is `O(k³)` where `k` is the regressor count - cheap compared to the meat formation.

### Tuning Parameters

| Parameter | Type | Default | Selection Method |
|-----------|------|---------|-----------------|
| `vcov_method` | str | `"hc0"` | Set to `"conley"` to activate. |
| `conley_coords` | tuple of 2 str | `None` | User specifies the two column names for lat/lon (or projected x/y). Required when `vcov_method="conley"`. |
| `conley_cutoff_km` | float | `None` (no default) | User-supplied. Conley does not provide a plug-in selector. Recommend a robustness sweep (3-5 values spanning the relevant economic-distance range). For Phase 1, error if not supplied. |
| `conley_kernel` | str | `"bartlett"` | `"bartlett"` evaluated on pairwise distance `d_ij/h` is the practitioner default, matching R `conleyreg` and Stata `acreg`; this radial 1-D form is a specialization of Conley's explicit 2-D separable PSD-guaranteed Bartlett (Eq 3.14, page 12) and is not formally PSD-guaranteed itself. `"uniform"` matches Conley's "truncated window" (page 11) and is also not PSD in general (footnote 11). Emit a warning under either kernel when the resulting meat has a materially negative eigenvalue. |
| `conley_metric` | str or callable | `"haversine"` | `"haversine"` for lat/lon (km); `"euclidean"` for projected coords (units = whatever the coord units are - so if coords are degrees, cutoff is in degrees); a callable `(coord_i, coord_j) -> float` for custom metrics (e.g., travel time, network distance). |

### Relation to Existing diff-diff Estimators
- **Composes with `compute_robust_vcov` in `diff_diff/linalg.py`**: Conley is a new value of `vcov_method` alongside `"hc0"`, `"hc1"`, `"cluster"`, `"crv1"`. The bread `(X'X)^{-1}` is unchanged; only the meat formation differs.
- **Reduces to HC0 when `conley_cutoff_km = 0` and no spatial ties**: with `K(d/0) = 1{d ≤ 0}` and all distinct units having `d_{ij} > 0`, the only nonzero terms are `i = j`, recovering the HC0 meat `Σ_i x_i ε̂_i² x_i'`. (Document this reduction; do NOT silently turn `cutoff=0` into HC0 - error out and tell the user to use `vcov_method="hc0"` directly.)
- **Differs from cluster-robust (`crv1`)**: cluster-robust uses a discrete group indicator `g_i = g_j` in place of `K(d_{ij}/h) = 1`. Conley with `conley_kernel="uniform"` and a cutoff that isolates exactly each cluster is the closest analog, but Conley is more general: it accommodates continuous-distance attenuation (Bartlett) and overlapping spatial neighborhoods (a unit can be "near" multiple others without belonging to a single cluster).
- **TWFE compatibility (Phase 1 scope)**: Conley replaces the meat in the standard sandwich. TWFE estimator `(X'X)^{-1} X'y` with absorbed fixed effects produces residuals `ε̂_i`; Conley's pairwise outer product over `(x_i ε̂_i, x_j ε̂_j)` is well-defined regardless of whether the fixed effects were absorbed or dummied. Phase 2 will extend to two-way space x time clustering (Cameron-Gelbach-Miller-style multi-way Conley).
- **Distance metrics**: paper is agnostic - Conley names "transportation costs" (Section 5 page 20), "physical distance" (page 2), "weather correlation" (page 3), "travel time" (page 2). Haversine vs euclidean is an applied-implementation choice not flagged in the paper. diff-diff's `conley_metric="haversine"` is the standard choice for lat/lon and matches Hsiang's MATLAB / Colella et al.'s `acreg`; `"euclidean"` is appropriate for projected coordinates.

---

## Gaps and Uncertainties

- **No 1-D pairwise kernel formula in the paper.** The paper only writes the 2-D product Bartlett (Equation 3.14, page 12). The applied-econ practitioner form `K(u) = max(0, 1 - |u|)` evaluated at `u = d_{ij}/h` is conventional but not derived in Conley 1999. Implementation should cite Conley 1999 for the framework, but credit the 1-D pairwise specialization to the downstream literature (Hsiang 2010, Colella et al. 2019). Page 18's general statement about `K(·,·)` "bounded continuous on [-1,1] x [-1,1]" with absolutely summable Fourier coefficients is the closest formal authority for kernels other than the explicit Bartlett product.

- **Bandwidth selection is left open.** Conley's only formal restriction is `L_M, L_N = o((MN)^{1/3})` (page 12). The empirical example (page 21) sweeps the cutoff over a coarse grid and reports robustness. There is no Andrews (1991) plug-in selector or cross-validation procedure in the paper. The implementing engineer must either expose `conley_cutoff_km` as a required user-supplied parameter (Phase 1 plan) or implement a practitioner heuristic separately (e.g., median nearest-neighbor distance times a multiplier) and document it as a diff-diff convenience, not a Conley-1999 result.

- **Haversine vs euclidean.** Conley works in R^2 with euclidean distance throughout (page 4 "Euclidean space, taken for the sake of exposition to be R^2"). The applied literature on country / county / household data routinely uses haversine on (lat, lon) - this is implicit in Hsiang (2010) and Colella et al. (2019). The paper does not address whether haversine satisfies the regularity conditions, but since haversine is just euclidean on the sphere and the mixing conditions are stated in terms of distance decay, the substantive content carries over for cutoffs small relative to the Earth's radius. Document this as an applied convention, not a theorem.

- **Ties in distance (`d_{ij} = 0`, `i ≠ j`).** Page 19 says the estimator stays consistent under E1-E2 but the PSD-by-construction guarantee fails. Practitioner workflow: the user must assign nearby-but-distinct coordinates (Conley's example: "may be sensible to arbitrarily assign nearby but distinct locations to observations within a city"). diff-diff Phase 1 should NOT silently jitter; instead, error or warn. Defer auto-jittering to Phase 2 if requested.

- **OLS is just-identified GMM (page 19, Section 5).** The paper's full GMM machinery covers overidentified moment conditions; Conley uses OLS in Section 5 to simplify the comparison vs HC0. For Phase 1 (TWFE OLS), the GMM `D_0 = (X'X)^{-1}` and `Ω = I` specializations are exactly the right form. The full GMM `Ω_τ` weighting matrix (Equation 3.3 page 7) is NOT relevant to diff-diff Phase 1; it would apply only if diff-diff added a 2SLS / GMM estimator (out of scope).

- **`λ = E W_s` factor in Equation 3.12.** In the lattice formulation, `C = λ^{-1} V` accounts for the fraction of lattice points actually sampled. In the practitioner pairwise form `Σ_{i,j} K(d_{ij}/h) X_i ε̂_i ε̂_j X_j'`, this factor is absorbed by the change of indexing (sum over actual observations rather than over lattice points). The implementing engineer should NOT multiply by `1/λ` in the pairwise form; this is already handled by summing over the realized sample.

- **PSD failure for both supported kernels under the radial 1-D specialization.** Two distinct sources:
  - *Uniform/truncated kernel* (footnote 11, page 11): Conley's exact wording: "This estimator will not always be PSD, unfortunately, since the spectral window corresponding to the step function space domain window (its Fourier transform) will be negative in some regions."
  - *Radial 1-D Bartlett* (the form diff-diff implements, matching R `conleyreg` / Stata `acreg`): Conley's explicit PSD-guaranteed Bartlett formula (Eq 3.14, page 12) is the 2-D **separable product window** `(1 - |j|/L_M)(1 - |k|/L_N)`, NOT the 1-D radial form on pairwise distance. The radial specialization is a practitioner convention (see "Pairwise (1-D) Bartlett" line above) that is not formally PSD-guaranteed.
  Implementation guidance: under either kernel, compute the eigenvalues of the meat after sandwich and if `min(eig) < -1e-12`, either (a) warn and proceed (matches `acreg` and `conleyreg`), (b) clamp to PSD via eigendecomposition + zero-floor, or (c) redirect to a separable 2-D product kernel (Phase 2 + space-time extension). Phase 1 plan: warn and proceed (option a) for both kernels to match downstream-tool expectations.

- **The empirical example uses cross-country growth regressions (page 20)**, NOT a DiD or panel setup. Conley does not work out the panel TWFE specialization in the paper. The diff-diff Phase 1 implementation extends Conley's machinery to TWFE OLS (which is a linear regression with absorbed fixed effects) - this is mechanically straightforward but the methodological extension warrants a citation to a downstream paper (e.g., Cameron-Miller 2015 review article, or Bester-Conley-Hansen 2011 spatial cluster bootstrap) in REGISTRY.md.

- **Sample size / boundary-effect caveats.** The proof of Proposition 3 (pages 26-31) relies on the boundary terms `Σ_{s_i ∈ Λ_τ - Λ_τ*} g(X_{s_i}; b)` shrinking faster than the interior. For finite samples on irregular regions (e.g., a country with concave coastline), the interior approximation may be less tight. The paper does not give a finite-sample correction. Phase 1 should pass `T_τ - k` (regressors-adjusted) as the divisor in any HC1-style finite-sample correction, mirroring HC0/HC1 conventions; HC0 (no correction) is the canonical Conley form per the paper.

- **Two-way (space x time) HAC**. Conley 1999 only treats cross-section. The diff-diff Phase 2 implementation (shipped 2026-05) extends to space × time on panel `MultiPeriodDiD` / `TwoWayFixedEffects`. **Phase 2 status update:** The implementation followed R `conleyreg`'s **block-decomposed sandwich**, NOT a multiplicative product kernel `K_space · K_time`. The total meat is additive: `XeeX = Σ_t (within-period spatial) + Σ_u (within-unit Bartlett serial, lag=0 excluded)`. The originally-hypothesized product-kernel form (per the original plan and Conley's 2-D spectral density discussion in Section 3.3) was empirically refuted by a spike against `conleyreg::time_dist.cpp` — `conleyreg`'s temporal contribution is a separate per-unit sandwich, not a Hadamard product. The block-decomposed form matches `conleyreg` at ~1e-14 across `lag_cutoff ∈ {0, 1, 2}` on the panel parity fixtures. Driscoll-Kraay (1998) and Hansen (2007) are the closest published references; cite `conleyreg` (Düsterhöft 2021, CRAN v0.1.9) source code for the exact algorithm.

- **No bandwidth-selection theorem.** Conley's bandwidth restriction (`o(n^{1/3})`) is purely a consistency rate; it gives no MSE-optimal rule. Andrews (1991) provides a plug-in selector for time series HAC under stronger assumptions; an analogous spatial plug-in does not appear in this paper. Modern alternatives (Bester-Conley-Hansen 2011 spatial cluster bootstrap; Müller 2014 worst-case bandwidth selection) are downstream developments not covered here. Document `conley_cutoff_km` as a user-required parameter with a robustness-sweep recommendation.

- **Acknowledgments + provenance** (page 23): Paper "is taken from my Ph.D. thesis at the University of Chicago. An earlier version circulated with the title 'Econometric Modelling of Cross Sectional Dependence.'" Lars Hansen, James Heckman, José Scheinkman acknowledged. NSF / Searle / Reid fellowships funded. The empirical example draws on the unpublished working paper Conley & Ligon (1995) "Economic distance, spillovers, and growth." This connects directly to the diff-diff Phase 3 spillover-regressor work — Conley & Ligon's notion of "spillover via economic distance" is the conceptual foundation for the bias-side spillover discussion in our Phase 3 plan.

- **Bolthausen (1982) CLT dependency.** The asymptotic-normality result (Proposition 2) leans on Bolthausen's central limit theorem for stationary mixing random fields on regular lattices (page 9; cited in proofs page 25). The Bolthausen reference is "On the central limit theorem for stationary mixing random fields" *Annals of Probability* 10, 1047-1050. For implementation correctness this is not load-bearing, but anyone porting the proofs (e.g., for a panel space-time extension) needs the Bolthausen technical inputs, which include rectangular sample regions (page 25, paragraph above mixing-coefficient definition `π(Λ_1, Λ_2)`). Footnote 9 (page 10) flags that "extension to non-rectangular `Λ_τ` is straightforward" but tedious.

- **Treatment of `K(0,0) = 1` in Equation 3.13.** The summation indices in Equation 3.13 are `j = 0, ..., L_M`, `k = 0, ..., L_N`, `m = j+1, ..., M`, `n = k+1, ..., N`. The `(j, k) = (0, 0)` term contributes `Σ_{m,n} K_{MN}(0,0) Y_{m,n} Y_{m,n}'` doubled (because of the symmetric `[Y Y' + Y' Y]` term), which is then de-duplicated by the explicit subtraction `-(1/T_τ) Σ Y_{m,n} Y_{m,n}'`. After cancellation the diagonal `i = j` contribution is exactly `Σ_i Y_i Y_i' = Σ_i x_i ε̂_i² x_i'` (the HC0 meat). The implementing engineer must reproduce this de-duplication carefully in the pairwise form: the formula `Σ_{i,j} K(d_{ij}/h) X_i ε̂_i ε̂_j X_j'` over **all** ordered pairs `(i, j)` (including `i = j`) automatically gives the right diagonal contribution `K(0) · X_i ε̂_i² X_i' = X_i ε̂_i² X_i'` (since `K(0) = 1`) and the right off-diagonal `K(d_{ij}/h) · X_i ε̂_i ε̂_j X_j'` for `i ≠ j` summed over `(i, j)` and `(j, i)` (both directions). No separate de-duplication needed in the pairwise form. Phase 1 unit test: assert that at `h → 0+` (no spatial ties), the Conley meat equals the HC0 meat to machine precision.
