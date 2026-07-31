# Clustered-variance conventions: measured inventory

> **Repo-internal methodology document** (excluded from the published docs
> build). This is the map behind the 3.9 variance-consolidation program. The
> table below is **generated** from the expected literals in
> tests/test_variance_conventions.py — run
> python -m tests.test_variance_conventions and paste the output here after
> any change that legitimately moves a cell. The parametrized tests assert
> those literals against live instrumentation (fast subset in the default
> suite, full sweep under -m slow), so a stale table fails CI on the
> expected-literal side, not silently.

## The measured matrix

Shared DGP: numpy.default_rng(7), 60 units x periods 1..6,
first_treat = [0, 3, 4, 5][unit % 4], treated = 1{ft > 0 and t >= ft},
grp = unit % 2 (time-invariant), post = 1{t >= 4},
y = N(0,1) + 0.3 unit + 0.2 t + 2 treated; n = 360, G = 60 under
cluster="unit". Each row's exact fit kwargs are in the test file — the
k values and SEs in any claim below come from the SAME fit (mixing panels
produced wrong figures three separate times while this inventory was drafted).

| surface | CR1 `k` (multiset) | tail df (multiset) | status | reason |
|---|---|---|---|---|
| `did_absorb_hc1_cluster_unit` | 7 | 294 | **legitimate** | K_reference (D2 fixed): k = 2 visible + rank(time given unit) = 5; matches reghdfe/fixest ssc(K.fixef='nested') |
| `did_fixed_effects_hc1_cluster_unit` | 7 | 294 | **legitimate** | K_reference (D1 fixed): 66 visible minus the 59 cluster-nested unit dummies -> identical SE to did_absorb (documented deviation from a literal explicit-dummy R comparison, which counts all 66) |
| `did_plain_hc1_cluster_unit` | 4 | 356 | **legitimate** | no absorbed FE: visible k is the whole design; nothing is omitted |
| `twfe_hc1_cluster_unit_time_post` | 3 | 298 | **legitimate** | K_reference (D2 fixed): 2 visible + rank(post given unit) = 1; matches fixest cluster arm at rel 0 (committed golden) |
| `wooldridge_hc1_within` | 15 | None, None, None, None, None, None, None, None, None, None | **defect** | CR1 k converged on K_reference (D2 fixed: 9 cells + T = 15, no intercept col -> +1 term; jwdid arms at ratio 1.0); tail df is still normal theory with no df_convention knob (PR C) |
| `sun_abraham_hc1` | 21 | 280, 280, 280, 280, 280, 280, 280, 280, 280, 280, 280, 280, 280, 280, 280, None, None, None, None, None, None, None, None | **defect** | CR1 k converged on K_reference (D2 fixed: 15 cells + 6, no intercept col; fixest sunab parity ~5e-15); D4 remains: residual df per cohort-period cell but normal theory on aggregates (PR C) |
| `stacked_did_hc1` | 6 | None | **legitimate** | L1: k_total is clubSandwich CR1S by construction (stacked_did.py pins vcovCR(type='CR1S') at atol=1e-10); normal-theory tail df is an open PR C question |
| `lpdid_pre2_post2` | 4, 4, 5, 5, 5, 6 | 59, 59, 59, 59, 59, 59 | **legitimate** | L2: G-1 tail df (Stata/fixest convention) — the convergence target |
| `mpd_absorb_hc1_cluster_unit` | 11 | 290, 290, 290, 290, 290, 290 | **legitimate** | K_reference: 6 visible + rank(time given unit) = 5; equals the fixed_effects form's 70 - 59 (MPD absorb/fixed_effects equivalence) |
| `mpd_fixed_effects_hc1_cluster_unit` | 11 | 290, 290, 290, 290, 290, 290 | **legitimate** | K_reference: 70 visible (incl. built-in period dummies, MPD's time-FE block) minus the 59 cluster-nested unit dummies = 11 — identical to the absorb form |
| `mpd_plain_hc1_cluster_time` | 7 | 348, 348, 348, 348, 348, 348 | **legitimate** | the NESTED orientation of the built-in period dummies: 12 visible minus their rank 5 under a time cluster (under-subtraction is caught here; the unit-cluster rows catch over-) |
| `lpdid_absorb_nested_cluster_grp` | 4, 4, 5, 5, 5, 6 | 1, 1, 1, 1, 1, 1 | **legitimate** | LPDiD absorb dummies nested in the cluster subtract their rank (adj -1 per horizon: region == grp here); _event_time stays counted (unit-level cluster does not nest time); G-1 tail df (L2) |
| `imputation_default` | — | None | **legitimate** | L3: BJS imputation variance, not the shared CR1 sandwich |
| `imputation_pretrends_event_study` | unpinned | unpinned | **defect** | pretrends lead regression: CR1 k converged on K_reference (D2 fixed); normal-theory tail df remains (PR C family) |
| `two_stage_default` | — | None | **legitimate** | L3: Gardner two-stage variance, not the shared CR1 sandwich |
| `callaway_santanna_default` | — | None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None | **legitimate** | L3: influence-function variance anchored to Stata csdid |

cr1_k is the sorted multiset of K_reference counts reaching the shared
clustered CR1 denominator — visible columns + the signed cluster_k_adjustment
(linalg._compute_robust_vcov_numpy with
vcov_type="hc1"; a clustered call in any other family fails the row, so a
surface cannot silently switch clustered family behind an unchanged design
width); — means the surface's *contract* is that it never calls it. tail_df is the multiset
of df values passed to safe_inference/safe_inference_batch
(None = normal theory). unpinned marks the one contract row whose test
asserts the shared CR1 IS reached but deliberately pins no exact values —
a literal there would be brittle configuration-detail (its expected_adjustment
IS pinned: +6, the [time, unit] increment on df_0). Captured under the
canonical Python backend; on adjusted clustered surfaces the Rust lanes apply
the same correction as an exact scalar rescale of the finished vcov
(vcov * (n-k)/(n-k-adj), <= 1 ulp on top of the default lane's <= 8e-15
cross-backend agreement; zero-adjustment surfaces keep bit-identical Rust
output).

## Fixed defects (3.9 consolidation program)

- **D1 — absorb= vs fixed_effects=: 10.35% SE split on the same model.
  FIXED (3.9, K_reference convergence).** Historically: identical ATT, se
  0.2414226781 (k=2) vs 0.2664071714 (k=66), ratio 1.103489 =
  sqrt((360-2)/(360-66)) exactly, in opposite directions from the reference.
  Both idioms now produce the identical K_reference = 7 on the audit panel
  (absorb: 2 + 5; full-dummy: 66 - 59 nested rank) and the identical SE
  0.2414226781 * sqrt((360-2)/(360-7)), pinned by
  test_d1_convergence_is_pinned. The equivalence is rank-based, so it holds
  under collinearity drops and on disconnected panels (kernel k is the design
  RANK; the two-nested-dim discriminator in TestKReferenceConvergence).
  The full-dummy side deviates from a LITERAL explicit-dummy R comparison
  (which counts all 66) — the labeled deviation note in REGISTRY
  "Deviation from R (clustered CR1, full-dummy fixed_effects= path)".
- **D2 — clustered CR1 never counted absorbed FE not nested in the cluster.
  FIXED (3.9, cluster_k_adjustment seam).** Historically anti-conservative:
  SEs understated by 0.10% (n=2500), 1.30% (n=200), 5.51% (n=60). The
  correction
  K_reference = explicit cols + (1 if no intercept col) + rank(non-nested FE | nested)
  (the +1 is the rank of the ABSORBED constant, so it exists only when FE are
  absorbed and X carries no intercept column) is computed by
  utils.absorbed_fe_cr1_k_increment and threaded keyword-only through
  solve_ols / compute_robust_vcov / LinearRegression.fit / wild_bootstrap_se.
  External anchors: Stata reghdfe 3.2.9 (via jwdid) at machine precision on
  all three committed mpdta arms AND at every rung of the committed G≈20..500
  subsample ladder (ratio 1.0, spreads ~1e-15..1e-14 — the ladder also probes
  the accounting itself: reghdfe df_a == increment - 1 per rung); R fixest
  0.14.2 cluster arms tightened to exact/1e-9. The non-nested term is the
  exact RANK given the nested set: on disconnected designs this deviates by
  one df per extra component from BOTH references (neither implements the
  exact composition — fixest's K.exact composes incoherently with its nested
  drop, reghdfe's pairwise correction skips nested-dropped dims), pinned as an
  exact sqrt((n-10)/(n-11)) SE ratio in the committed
  fixest_cr1_nonnested_golden.json / reghdfe_kref_golden.json arms, while the
  nothing-nested crossed-cluster arms match both references at machine
  precision. Fail-closed: n_eff - k <= 0 (visible saturation), n_eff - k_inf
  <= 0, or k_inf <= 0 each yield the all-NaN vcov on every backend.
- **D3 — absorbed rank assumed independent, connected FE dimensions.**
  sum(levels - 1) over-counted on disconnected panels (true rank
  sum(levels) - C) and on hierarchical specs
  (absorb=["state", "state_year"]: true 29, old count 34). **Fixed** by
  diff_diff.utils.absorbed_fe_rank (this PR): two-way rank from the
  bipartite level graph's connected components; N >= 3 keeps sum(levels-1)
  with the limitation documented (over-counts for duplicated/nested triples —
  tracked in TODO.md). **External anchor:** the exact rank matches
  ``fixest::ssc(K.exact = TRUE)`` at machine precision on the committed
  hierarchical golden (``benchmarks/data/fixest_kexact_golden.json``); fixest's
  DEFAULT ``K.exact = FALSE`` reproduces the old naive count, so this is a
  documented deviation from the R default (see the REGISTRY absorbed-FE note
  and ``tests/test_variance_conventions.py::TestFixestKExactParity``).
- **D4 — SunAbraham reports two tail-df conventions inside one fit** (residual
  df on per-cell inference, normal theory on aggregates — visible in its row's
  multiset). Fix: PR C.

## Legitimate differences (declared exceptions)

- **L1 — StackedDiD's k_total**: its design genuinely is a Q-weighted
  full-dummy lm; pinned to clubSandwich::vcovCR(type="CR1S") at
  atol=1e-10. CR1S is a real second convention, correct by construction.
- **L2 — LPDiD's G-1 tail df**: the Stata/fixest convention, and the only
  surface where it is the default. It is the convergence target for PR C, not a
  defect.
- **L3 — CallawaySantAnna / TwoStageDiD / ImputationDiD (default)**: different
  variance theory (influence functions / two-stage / BJS imputation), never the
  shared CR1 sandwich. CS is anchored to Stata csdid outright.
  **ImputationDiD is conditional**: its pretrends=True +
  aggregate="event_study" lead regression DOES run the shared clustered CR1
  and carries the K_reference increment there (+6, the [time, unit] no-intercept
  increment on df_0 — pinned via expected_adjustment on its matrix row).
- **L4 — hc2/hc2_bm** (leverage / Satterthwaite DOF — no CR1 factor),
  **survey TSL** (n_PSU - n_strata over the full design), and **Wooldridge
  cohort_trends full-dummy** (documented opt-in landing on the L1
  convention). **conley** is out of this matrix by decision: the spatial-HAC
  family applies no CR1 finite-sample factor, so it has no cell on the axis this
  matrix measures.

## Tail-df landscape (PR C's input)

Three conventions are live: normal theory (Wooldridge, StackedDiD — no
df_convention knob), residual n - K_full (DiD/MPD/TWFE default;
df_convention="cluster" opts into G-1), and G-1 (LPDiD, hardcoded).
At |t| = 2 normal theory understates the t(G-1) p-value by 24.2% at
G=20, 13.3% at G=40, 1.2% at G=500 — larger than the D2 SE gap was before its
fix. PR C decisions:
(1) extending the knob to Wooldridge/SunAbraham/StackedDiD/ImputationDiD needs
NEW ledger rows (M-004/M-005/M-006 cover only DiD/TWFE/LinearRegression);
(2) the two-value knob cannot express normal theory, so either a third value
("normal") keeps 3.9 additive, or a documented default change ships with its
own ledger rows.
