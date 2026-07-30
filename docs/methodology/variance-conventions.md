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
| `did_absorb_hc1_cluster_unit` | 2 | 294 | **defect** | D2: CR1 k omits absorbed FE not nested in the cluster (time) |
| `did_fixed_effects_hc1_cluster_unit` | 66 | 294 | **defect** | D1: same model as did_absorb yet k=66 vs 2 -> SEs differ 10.35%; full-dummy k also counts the cluster-nested unit FE the references drop |
| `did_plain_hc1_cluster_unit` | 4 | 356 | **legitimate** | no absorbed FE: visible k is the whole design; nothing is omitted |
| `twfe_hc1_cluster_unit_time_post` | 2 | 298 | **defect** | D2 (within-transform k_visible); tail df is residual n-K_full |
| `wooldridge_hc1_within` | 9 | None, None, None, None, None, None, None, None, None, None | **defect** | D2 (k_visible=cells only) + normal-theory tail df with no df_convention knob |
| `sun_abraham_hc1` | 15 | 280, 280, 280, 280, 280, 280, 280, 280, 280, 280, 280, 280, 280, 280, 280, None, None, None, None, None, None, None, None | **defect** | D2 + D4: residual df per cohort-period cell but normal theory on aggregates |
| `stacked_did_hc1` | 6 | None | **legitimate** | L1: k_total is clubSandwich CR1S by construction (stacked_did.py pins vcovCR(type='CR1S') at atol=1e-10); normal-theory tail df is an open PR C question |
| `lpdid_pre2_post2` | 4, 4, 5, 5, 5, 6 | 59, 59, 59, 59, 59, 59 | **legitimate** | L2: G-1 tail df (Stata/fixest convention) — the convergence target |
| `imputation_default` | — | None | **legitimate** | L3: BJS imputation variance, not the shared CR1 sandwich |
| `imputation_pretrends_event_study` | unpinned | unpinned | **defect** | pretrends lead regression runs the shared clustered CR1 with k_visible |
| `two_stage_default` | — | None | **legitimate** | L3: Gardner two-stage variance, not the shared CR1 sandwich |
| `callaway_santanna_default` | — | None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None | **legitimate** | L3: influence-function variance anchored to Stata csdid |

cr1_k is the sorted multiset of visible column counts reaching the shared
clustered CR1 denominator (linalg._compute_robust_vcov_numpy with
vcov_type="hc1"; a clustered call in any other family fails the row, so a
surface cannot silently switch clustered family behind an unchanged design
width); — means the surface's *contract* is that it never calls it. tail_df is the multiset
of df values passed to safe_inference/safe_inference_batch
(None = normal theory). unpinned marks the one contract row whose test
asserts the shared CR1 IS reached but deliberately pins no exact values —
a literal there would be brittle configuration-detail (last measured:
k=3, normal-theory tail df on all 8 event-study leads/lags). Captured under the canonical Python backend; Rust
and Python agree to <= 8e-15 on every surface because both implement the same
conventions today.

## Defects (scheduled: 3.9 consolidation program)

- **D1 — absorb= vs fixed_effects=: 10.35% SE split on the same model.**
  Identical ATT, se 0.2414226781 (k=2) vs 0.2664071714 (k=66), ratio
  1.103489 = sqrt((360-2)/(360-66)) exactly. The absorb-side deviation from
  fixest is documented (REGISTRY "Known limitation (deviation from fixest)",
  DEFERRED.md); the *user-facing consequence* — two documented-equivalent kwargs
  disagreeing by 10%, in opposite directions from the reference — was not, until
  this inventory. Fix: PR B converges both on K_reference.
- **D2 — clustered CR1 never counts absorbed FE not nested in the cluster.**
  k_visible in the denominator; _absorbed_fe_vcov_scale exists but is
  gated on cluster_ids is None. Anti-conservative: SEs understated by 0.10%
  (n=2500), 1.30% (n=200), 5.51% (n=60). The correction
  K_reference = explicit cols + (1 if no intercept col) + rank(non-nested FE | nested)
  reproduces Stata reghdfe 3.2.9 (via jwdid) to ~1e-15 on three arms and
  R fixest 0.14.2 to ~1e-12 on two, and retrodicts the two measured-but-unpinned
  subsample rungs (predicted 1.028016 vs recorded 1.0280 at G=20; 1.013210 vs
  1.0132 at G=40). Fix: PR B.
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
  and inherits D2 there (its own matrix row).
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
G=20, 13.3% at G=40, 1.2% at G=500 — larger than the D2 SE gap. PR C decisions:
(1) extending the knob to Wooldridge/SunAbraham/StackedDiD/ImputationDiD needs
NEW ledger rows (M-004/M-005/M-006 cover only DiD/TWFE/LinearRegression);
(2) the two-value knob cannot express normal theory, so either a third value
("normal") keeps 3.9 additive, or a documented default change ships with its
own ledger rows.
