# Stata parity benchmarks

Stata golden generators live here (six arms: LPDiD `teffects ra`, LPDiD
non-absorbing `lpdid`, ImputationDiD leave-one-out, ETWFE/CS, reghdfe
K_reference, LWDiD); the
pattern mirrors `benchmarks/R/` (a `generate_*` script writes a committed
golden JSON that a skip-guarded test reads, so CI never needs Stata). Stata
is node-locked single-user, so — exactly like the R arm — goldens are
committed and only regenerated locally.

Locating the binary (macOS, StataSE 19; **not** on `PATH` by default):

```
/Applications/Stata/StataSE.app/Contents/MacOS/stata-se
```

---

# `teffects ra` parity for the LP-DiD regression-adjustment SE

`benchmarks/stata/generate_lpdid_ra_golden.do` produces
`benchmarks/data/lpdid_ra_stata_golden.json`, consumed by
`tests/test_lpdid_ra_stata_parity.py` to verify that diff-diff's LP-DiD
regression-adjustment (RA) standard error (Dube, Girardi, Jorda & Taylor 2025)
matches Stata `teffects ra ... atet vce(cluster)` across the 7 event-study
horizons.

## Why Stata `teffects`

The RA covariate path reports an influence-function cluster variance with **no
finite-sample factor**. **No R package computes it** — `alexCardazzi/lpdid` uses
direct covariate inclusion, not RA — so there is no runnable R analogue. The
canonical reference is Stata `teffects ra ... atet vce(cluster)`, which the LP-DiD
authors themselves invoke (paper footnote 9). `teffects` is a **native** Stata
command (no SSC package), so `version 19` in the generator fully pins its
numerical behavior; there is no third-party package version to track.

This arm converts REGISTRY `## LPDiD` Deviation 2 from an *inference* (the
no-finite-sample-factor convention was argued from degrees-of-freedom comments in
the authors' `.do` files) into a *measurement*: the library RA IF SE matches
`teffects` to ~1e-16 at every horizon (same machine / BLAS).

## What the generator does

It reads (does **not** regenerate) `benchmarks/data/lpdid_test_panel.csv` — whose
sole owner is `benchmarks/R/generate_lpdid_golden.R` — and **independently
reconstructs** each horizon's clean sample by porting the R `prep` + `clean_h`
recipe (`generate_lpdid_golden.R:97-123`) to `tsset` + `L.`/`F.` time-series
operators. This gives three independent sample constructions (Python / R / Stata);
a port bug is very likely to be caught before it could masquerade as an SE finding,
because the Python test's point gate (Stata ATET vs the R-anchored `ra_cov[h][0]`) and
sample-shape gate (`(e(N), e(N_clust))` vs the library's `(n_obs, n_clusters)`) would
fail first.

`tsfill` is load-bearing: it mirrors R `fill_gaps()` so the absorbing `treat`
recompute and the long differences see the completed calendar grid across the
deliberate interior gap at `(unit==60, time==7)`.

The per-horizon command is:

```stata
teffects ra (Dy x i.time) (tdiff), atet vce(cluster unit)
```

(The authors' footnote-9 syntax omits the covariate; the RA *covariate* path adds
`x` to the outcome model, as here.)

## Regenerating

```bash
# from the repo root
/Applications/Stata/StataSE.app/Contents/MacOS/stata-se -b do \
    benchmarks/stata/generate_lpdid_ra_golden.do

# batch mode ALWAYS exits 0 — verify the log has no Stata errors:
grep -E '^r\([0-9]+\);' generate_lpdid_ra_golden.log   # must print nothing
```

The output JSON is **committed** so CI doesn't need Stata. Only re-run when the
committed panel changes (then also refresh `RA_SE_PIN` in
`tests/test_methodology_lpdid.py`) or the horizon set changes.

## Skip behavior

`tests/test_lpdid_ra_stata_parity.py` calls `pytest.skip(...)` when the golden (or
the shared panel / R golden it cross-checks against) is absent, so CI passes
without Stata. The estimator itself is exercised independently by
`tests/test_methodology_lpdid.py` and `tests/test_lpdid.py`.

## JSON schema

```json
{
  "meta": {
    "estimator": "LPDiD regression-adjustment (RA) SE - Stata teffects ra atet",
    "generator": "benchmarks/stata/generate_lpdid_ra_golden.do",
    "source_panel": "benchmarks/data/lpdid_test_panel.csv",
    "point_anchor": "benchmarks/data/lpdid_golden.json ra_cov[h][0]",
    "stata_edition": "SE",
    "cmd": "teffects ra (Dy x i.time) (tdiff), atet vce(cluster unit)",
    "se_convention": "...",
    "note": "...",
    "stata_version": 19.0,
    "pre_window": 3,
    "post_window": 4
  },
  "ra_se": {
    "0":  {"att": <float>, "se": <float>, "N": <int>, "G": <int>},
    "...": "post {0..4} then pre {-2,-3}; h=-1 is the omitted reference"
  }
}
```

Values are written at `%21.17g` (round-trip-exact IEEE-754 double). `N` and `G`
(`e(N)` / `e(N_clust)`) are emitted so the Python gate can assert the Stata clean
sample matches the library's realized size and cluster count — which, together with
the point (1e-10) and SE (~1e-16) agreement, strongly corroborates the same clean
sample. No timestamp — the golden regenerates byte-identically.

---

# `lpdid` parity for the LPDiD non-absorbing modes

`benchmarks/stata/generate_lpdid_nonabsorbing_golden.do` produces
`benchmarks/data/lpdid_nonabsorbing_stata_golden.json`, consumed by
`tests/test_lpdid_nonabsorbing_stata_parity.py`. Unlike the `teffects` arm above, this
arm runs the authors' SSC **`lpdid`** package (Busch & Girardi, in collaboration with
Dube, Jorda & Taylor) **end-to-end** — the package builds its own clean samples — on
the committed `benchmarks/data/lpdid_nonabsorbing_panel.csv` (owner:
`benchmarks/R/generate_lpdid_golden.R`; 60 units x 14 periods, balanced, gap-free).
First external anchors: the non-absorbing **reweighted SE** (previously pinned-only
via `RW_SE_PIN`), the non-absorbing **pooled windows** (points and SEs), and the
Eq. 12 reweighted point.

## Option-to-mode mapping

| Stata `lpdid` | diff-diff |
|---|---|
| `nonabsorbing(, firsttreat notyet)` | `LPDiD(non_absorbing="first_entry")` (Eq. 12) |
| `nonabsorbing(L)` | `LPDiD(non_absorbing="effect_stabilization", stabilization_window=L)` (Eq. 13) |

## Parity scoping and the convention-neutral subsample

Eq. 12 agrees on EVERY surface of the full panel (all event-study horizons incl.
placebos + both pooled windows, vw and rw; att ~1e-14/1.3e-8, SE ~6e-17/2.3e-9).
Eq. 13 agrees at post horizons + pooled post only, and only on a 47-unit subsample,
because the package's sample construction differs on three measured conventions
(REGISTRY `## LPDiD` Deviation 4): (1) missing-lag boundary semantics admit
always-treated units as early-`t` controls (paper-silent surface); (2) its
switch-free lag window effectively requires `L+1` untreated periods before re-entry
where the paper's Eq. 13 states a levels condition over `[t-L, t-1]` (the package is
stricter than the paper); (3) placebo samples are built by recursive lagged
intersection of its `CCS_0` rather than the library's backward window (paper-silent).
The subsample removes what (1)+(2) admit differently — always-treated units
(`min(treat)==1`: 31–40) plus exact-`L`-respell units (any `t` with
`dD_t==1 & dD_{t-L}==-1`: 24, 25, 27) — making the Eq. 13 post samples row-identical
(658 rows). Convention (3) is not neutralized by THIS subsample (a dedicated
late-entry/never-treated subsample could in principle align the two placebo
constructions — a possible follow-up), so Eq. 13
placebo/pooled-pre rows and the Eq. 13 full-panel event study are recorded as
measured DIVERGENCE documentation (att + obs only) and gated as such
(divergence-floor + sample-size-mismatch asserts), never as parity.

## Regenerating

```bash
# once: install lpdid + deps (boottest, egenmore, listreg) from SSC
/Applications/Stata/StataSE.app/Contents/MacOS/stata-se -b do benchmarks/stata/requirements.do
# from the repo root
/Applications/Stata/StataSE.app/Contents/MacOS/stata-se -b do \
    benchmarks/stata/generate_lpdid_nonabsorbing_golden.do
# batch mode ALWAYS exits 0 — verify the log has no Stata errors:
grep -E '^r\([0-9]+\);' generate_lpdid_nonabsorbing_golden.log
python -m json.tool benchmarks/data/lpdid_nonabsorbing_stata_golden.json > /dev/null
```

The generator fails closed (`exit 111`) if any run-time dependency is missing and
records every one in `meta.ssc_versions` (guard set == version set; both guarded
egenmore files carry their own entry — `egenmore` via `_gfilter.ado`, the file the
pooled spec executes, and `egenmore_gclsst` via `_gclsst.ado`, lpdid's startup
which-check). Each version string also embeds the ado file's `checksum:`/`len:` so
a same-version-string upstream replacement still moves the drift metadata.

## JSON schema

```json
{"meta": {"...provenance...", "mapping": "...", "drop_rule": "...",
          "dropped_units": [...], "source_sha256": "...", "ssc_versions": {...}},
 "first_entry":     {"vw": {"es": {"-3".."4": {"att", "se", "N"}},
                            "pooled": {"pre": {...}, "post": {...}}}, "rw": {...}},
 "effect_stab_sub": {"n_rows": 658,
                     "vw": {"es": {"post rows: {att, se, N}; pre rows: {att, N}"},
                            "pooled": {"pre": {"att", "N"}, "post": {"att", "se", "N"}}},
                     "rw": {...}},
 "effect_stab_full_vw": {"es": {"0".."4": {"att", "N"}}}}
```

Divergent rows deliberately omit `se` — every recorded numeric value is covered by a
Python gate, and no gate anchors those SEs. No timestamp — byte-identical
regeneration under fixed Stata + fixed installed SSC versions.

---

# `did_imputation, leaveout` parity for ImputationDiD LOO SE

`benchmarks/stata/generate_imputation_loo_golden.do` produces
`benchmarks/data/didimputation_loo_stata_golden.json`, consumed by
`tests/test_imputation_loo_stata_parity.py` to verify that
`ImputationDiD(leave_one_out=True)` — the Borusyak-Jaravel-Spiess (2024) Supplementary
Appendix A.9 finite-sample variance refinement — matches Stata `did_imputation ...,
leaveout` at the overall ATT and all 6 event-study horizons.

## Why Stata `did_imputation`

The A.9 leave-one-out (LOO) variance has **no runnable R reference** — R
`didimputation` omits LOO entirely — so the library LOO SE was validated only by an
internal psi-identity + hand-calc + MC coverage. The authors' own Stata `did_imputation`
(Borusyak) ships the same option (`leaveout`); this arm turns it into a measured anchor.

## SSC dependence (vs the native-`teffects` LPDiD RA arm)

Unlike the native-`teffects` LPDiD RA arm, `did_imputation` is an SSC package with a
dependency chain `did_imputation → reghdfe → require + ftools`, none pinned by
`version 19`. (The ETWFE/CS, LWDiD and LPDiD non-absorbing arms are SSC-dependent
in the same way.) The generator does **not** install them — run
`benchmarks/stata/requirements.do` once first — and it records each package's version
in `meta.ssc_versions` (the `*!` ado header line) so drift is detectable. Byte-identical
regeneration is therefore scoped to a fixed Stata + fixed installed SSC versions.

## What the generator does

No clean-sample reconstruction: `did_imputation` consumes the raw committed R-arm panel
`didimputation_test_panel.csv` (180 units, cohorts {3,5} + never-treated); the mappings
are `Ei = first_treat` (missing for never-treated) and `K = time - Ei` (the relative
time grouped on under `avgeffectsby(K)`). Every partition is pinned **explicitly**:
`avgeffectsby(Ei t)` (== library `aux_partition="cohort_horizon"`, the default arm),
`avgeffectsby(Ei)` (== `"cohort"`), `avgeffectsby(K)` (== `"horizon"`). `Ei t` is also
`did_imputation`'s current default, but pinning each keeps the validation estimand
self-describing and robust to a future default change. The default arm's per-horizon
command is:

```stata
did_imputation y unit time Ei, horizons(0/5) leaveout avgeffectsby(Ei t) cluster(unit)
```

## Coarser-partition variants + unbalanced subsample

The `variants` block anchors the coarser `aux_partition` values on the same balanced
panel: `horizon` at the overall AND horizons 0..5 (its per-horizon SEs genuinely differ
from the default at h=0..3 and coincide at the single-cohort horizons h=4,5), `cohort`
at the overall only — because on a BALANCED panel the cohort partition is an
**arithmetic identity** with the default (only `v != 0` rows contribute per group; the
generator asserts the degeneracy at `reldif < 1e-6`, observed ~1e-12, and the Python
gate 7 pins it committed-vs-committed at 1e-9). The `unbalanced` block is where
`cohort` genuinely bites: a deterministic subsample (`drop if mod(unit,4)==0 &
time>=6`, N=1305 — reconstructed identically in pandas by the test; the row count alone
does not identify the subsample, the numeric parity gates do) re-runs all three
partitions at the overall, with the cohort LOO SE measuring ~23% above the default's.
The unbalanced point pin is **library-anchored** (no R anchor exists for the subsample;
informational — the Python test is authoritative). Under `leave_one_out=True` the
`horizon` partition's SE can be *smaller* than the default's (parity panel h=2), so
"coarser ⇒ more conservative" is typical, not guaranteed — recorded in the REGISTRY
ImputationDiD partition note.

Agreement is cross-implementation, not bit-identical (`did_imputation` goes through
`reghdfe`, the library through its own sparse IF solver): the **SE** agrees to ~1e-9 and
the **point** to ~2e-8; the parity test gates at `abs=1e-7`. Both the LOO and the non-LOO
cluster SE are emitted; the non-LOO SE additionally three-way-confirms against the R
golden (`didimputation_golden.json`).

**Panel-coupling caveat:** `didimputation_test_panel.csv` is owned by the R generator
(`benchmarks/R/generate_didimputation_golden.R`); if it is ever regenerated, the R golden
AND this Stata golden must both be regenerated.

## Regenerating

```bash
STATA=/Applications/Stata/StataSE.app/Contents/MacOS/stata-se
$STATA -b do benchmarks/stata/requirements.do            # one-time SSC install
$STATA -b do benchmarks/stata/generate_imputation_loo_golden.do
grep -E '^r\([0-9]+\);' generate_imputation_loo_golden.log   # must print nothing
```

## JSON schema

```json
{
  "meta": { "...": "...", "avgeffectsby": "default arm: Ei t (== cohort_horizon)",
            "avgeffectsby_variants": "Ei == cohort, K == horizon (K = t - Ei)",
            "ssc_versions": {"did_imputation": "...", "reghdfe": "...", "...": "..."} },
  "overall": {"att": <f>, "se": <LOO>, "se_nonloo": <f>, "N": <int>},
  "event_study": { "0": {"att": <f>, "se": <LOO>, "se_nonloo": <f>}, "...": "..." },
  "variants": {
    "cohort":  {"overall": {"att": <f>, "se": <LOO>, "se_nonloo": <f>}},
    "horizon": {"overall": {"...": "..."}, "event_study": {"0": {"...": "..."}, "...": "..."}}
  },
  "unbalanced": {
    "drop_rule": "mod(unit,4)==0 & time>=6", "n_rows": 1305,
    "cohort_horizon": {"att": <f>, "se": <LOO>, "se_nonloo": <f>},
    "cohort": {"...": "..."}, "horizon": {"...": "..."}
  }
}
```

---

# `jwdid` / `csdid` parity for the ETWFE and Callaway-Sant'Anna ATT(g,t)

`benchmarks/stata/generate_etwfe_cs_golden.do` produces
`benchmarks/data/etwfe_cs_stata_golden.json`, consumed by
`tests/test_etwfe_cs_stata_parity.py`. It anchors **both** staggered estimators
on the genuine `mpdta` panel: `WooldridgeDiD` against `jwdid` (Rios-Avila's
Wooldridge ETWFE) and `CallawaySantAnna` against `csdid`.

## Why this arm

`tests/test_wooldridge.py` asserted that ETWFE ATT(g,t) **equals** CS ATT(g,t)
within `5e-3`. That is false on real data — at `(g=2007, t=2007)` they differ by
`0.0171` (`-0.0431` vs `-0.0261`). The assertion only ever passed because
`load_mpdta()` was silently substituting a synthetic, effect-homogeneous DGP
when its source URL 404'd (issue #722), and on that DGP the two estimators do
coincide.

Stata's `jwdid` and `csdid` reproduce the **same** disagreement, which is what
establishes it as a property of the estimators rather than a bug in either. So
one self-referential cross-check that validated nothing was replaced by two
external anchors, with the ETWFE-vs-CS gap recorded rather than asserted away.

## What it measures

- **Point estimates** — both estimators match their reference to `atol=1e-6`
  (observed ~3e-8, i.e. Stata's log-output rounding) on all 7 post-treatment cells.
- **CS SEs** — match `csdid` outright (`rtol=1e-5`).
- **All-eventually-treated cell set and row count** (`jwdid_alltreated`) — the
  same panel with `first_treat == 0` dropped: 191 units, 955 rows. `jwdid`
  succeeds and estimates on 764 of them, because with no never-treated group the
  last cohort (2007) becomes the reference and the fully-treated periods carry
  no identified ATT (W2025 Section 5.4). It reports only the smaller `N`. The
  library computes the identical cell set
  (`(2004,2004), (2004,2005), (2004,2006), (2006,2006)`) on the identical row
  count, with ATTs matching to ~1e-15 — and warns about the reduction rather
  than passing it over in silence. `n` and `n_units` are serialized because the
  row count is the finding, not incidental.
- **ETWFE SEs** — match `jwdid` (reghdfe) at machine precision under the 3.9
  `K_reference` convergence. The historical uniform gap (1.0280 at G=20,
  1.0132 at G=40, 1.00264 at G=191, 1.0010 at G=500) was defect D2: the
  library's clustered CR1 factor counted only the visible treatment-cell
  columns, omitting the absorbed FE not nested in the unit cluster
  (`K_reference = cells + T` on this no-intercept within design). All arms now
  gate the SE ratio at 1.0 (`rtol=1e-9`; measured spreads ~1e-14..1e-15).
- **The subsample `ladder` block** — jwdid re-run on deterministic rosters
  (the first N units per `first.treat` cohort by ascending `countyreal`,
  N ∈ {5, 10, 20, 40, 80, 200, 500} → G ∈ {20, 40, 80, 140, 220, 391, 500}),
  each rung storing reghdfe's df accounting (`df_a` + its
  initial/nested/redundant decomposition, `rank`, `df_r`) alongside per-cell
  `att`/`se`. This gates the few-cluster behavior (where the historical gap
  was largest, ~2.8% at G=20) and doubles as the K-accounting probe: the
  consuming test asserts `df_a == absorbed_fe_cr1_k_increment − 1` at every
  cluster count. See the REGISTRY `## WooldridgeDiD (ETWFE)` hc1 note and
  `docs/methodology/variance-conventions.md`.

## Input panel

Reads (does **not** regenerate) `benchmarks/data/mpdta_stata_panel.csv` — the
upstream `mpdta.csv` at SHA-256 `2283bea1…3167`, the same digest pinned in
`diff_diff/datasets.py`. The panel is **committed rather than fetched** so this
arm never depends on network availability; network dependence is precisely the
failure mode that produced the false assertion. Both the generator and the
Python test assert the digest, so a swapped panel cannot silently retarget the
parity.

---

# `reghdfe` anchor for the clustered CR1 `K_reference` on a disconnected panel

`benchmarks/stata/generate_reghdfe_kref_golden.do` produces
`benchmarks/data/reghdfe_kref_golden.json`, consumed by
`tests/test_variance_conventions.py::TestReghdfeKReferenceParity`. The DGP is
**deterministic (no RNG)** — integer formulas the Python test rebuilds verbatim
— so no data is embedded or shipped: a disconnected two-way panel (units 0-9
observed in periods 0-4, units 10-19 in periods 5-9; C=2 components, span rank
`U + T − C = 28`).

Two arms:

- **`cross_cluster`** (cluster crosses both FE dims → nothing nested):
  reghdfe's pairwise dof method computes the exact span rank (`df_a = 28`,
  denominator `N − 29`) and the library matches at ~1e-17, agreeing with
  fixest `ssc(K.fixef="full", K.exact=TRUE)` (see
  `benchmarks/R/generate_fixest_cr1_nonnested_golden.R`). This is the parity
  anchor for the exact non-nested RANK term.
- **`unit_cluster`** (unit FE nested in the cluster): a DOCUMENTATION arm —
  reghdfe counts the nested-remainder approximately (`df_a = T − 1 = 9`,
  implied K = 11; its pairwise correction skips pairs containing a
  nested-dropped dim) where the library uses the exact remainder given the
  nested span (K = 10). No external reference implements that composition, so
  the test pins the deviation exactly: `se_reghdfe / se_library ==
  sqrt((N−10)/(N−11))`. On CONNECTED designs the two coincide — the jwdid
  subsample ladder above pins machine-precision agreement at every G.

# LWDiD parity vs the authors' `lwdid` package

`benchmarks/stata/generate_lwdid_golden.do` produces
`benchmarks/data/lwdid_stata_golden.json`, consumed by
`tests/test_methodology_lwdid.py` (import-skip-gated until `diff_diff.lwdid`
lands via PR #588) and schema-validated on main by the ungated
`tests/test_lwdid_stata_golden_schema.py`.

## Why the authors' `lwdid`

Printed-table goldens (LW 2026 Table 3, LW 2025 Tables A4/A5) only reach
three decimals and don't cover the randomization-inference convention or the
multiplier-bootstrap SEs at all. The authors' SSC `lwdid` (Hur, Lee &
Wooldridge) is the reference implementation; this arm pins it at full
precision: Prop 99 small-N ATT/SE + RI p at 100k reps (both rollings),
castle-doctrine `tau_omega` ATT/SE (both rollings), and the six Walmart
large-N event-study configs (per-r WATT points + B=9,999 multiplier-bootstrap
SEs, `set seed` pinned). It is an SSC arm: `meta.ssc_versions` records the
verbatim `lwdid` version line (fail-closed capture — the generator aborts if
no version line is found), and `capture which lwdid` exits 111 if the package
is missing (`requirements.do` installs it).

## Inputs and their integrity

Prop 99 / Walmart come from the loader cache
(`~/.cache/diff_diff/datasets/{prop99,walmart}.dta` — the authors' SSC
ancillary files). Loader success does NOT guarantee the on-disk bytes (cache
writes are best-effort), so the regeneration recipe hashes the cache files
directly. Castle uses the committed
`benchmarks/data/real/castle_lw_subset.csv`; note `set type double` makes
`import delimited` read it at double precision (a float import shifts the
8th decimal — the committed golden is the double-precision read, matching
Python). In-`.do` gates: `confirm numeric variable` + row-count asserts per
dataset. Measured-value smoke gates cover the SMALL-N blocks only (prop99
att + detrend RI p; castle att/se); the Walmart block's gates are
schema/cardinality/nonmissing (one row per event time, no missing cells) -
its value anchoring lives in the Python parity tests.

## Regenerating

```bash
# 1. fail-closed warm-up: loaders genuine AND on-disk cache bytes SHA-verified
python -c "from diff_diff import load_prop99, load_walmart; \
  import hashlib, pathlib; \
  dfs = {'prop99': load_prop99(), 'walmart': load_walmart()}; \
  assert all(df.attrs.get('source') != 'synthetic_fallback' for df in dfs.values()); \
  pins = {'prop99': '16c3ac1da351788817433fc890ec2f502a8bdfcb46cbc8d693653330e71d5a65', \
          'walmart': 'cf1b2d4aee396f0858fb315d0413ff82359e16e84dad68ce1d9be51e8c3e35ac'}; \
  cache = pathlib.Path.home() / '.cache' / 'diff_diff' / 'datasets'; \
  [1/0 for n, w in pins.items() if hashlib.sha256((cache / (n + '.dta')).read_bytes()).hexdigest() != w]"
# 2. generate (about 15 minutes; six B=9,999 bootstrap runs dominate)
STATA=/Applications/Stata/StataSE.app/Contents/MacOS/stata-se
$STATA -b do benchmarks/stata/requirements.do            # one-time SSC install
$STATA -b do benchmarks/stata/generate_lwdid_golden.do
grep -E '^r\([0-9]+\);' generate_lwdid_golden.log        # must print nothing
```

## JSON schema

```json
{
  "meta": {"ssc_versions": {"lwdid": "version 2.4.2 ..."}, "bootstrap_scheme": "...",
           "control_pool": {...}, "datasets": {...}, "stata_version": 19,
           "rireps": 100000, "riseed": ..., "bootstrap_reps": 9999, "bootstrap_seed": ...},
  "prop99":  {"demean": {"att": ..., "se": ..., "p_ri": ...}, "detrend": {...}},
  "castle":  {"demean": {"att": ..., "se": ...}, "detrend": {...}},
  "walmart": {"<rolling>_<method>__<outcome>": {
      "watt": {"-22": [point, se], ..., "13": [point, se]},
      "overall": {"Pre_avg": [point, se], "Post_avg": [point, se]}}, ...}
}
```

Stata missing values are emitted as JSON `null` (the `_jnum` writer has an
explicit missing branch).

## Known constraints

- **Batch mode always exits 0**, even on a hard error (`r(NNN);`). Never trust the
  shell exit code — parse the `.log` for `^r\([0-9]+\);`. The LPDiD (RA arm 1e-8;
  non-absorbing arm 1e-6), ImputationDiD (1e-6) and LWDiD generators (1e-10/1e-6 on
  its SMALL-N blocks; the LWDiD Walmart block gates schema/cardinality/nonmissing
  only) also run informational in-`.do` smoke gates that surface as `r(9);` on a
  gross bug; the Python parity tests are authoritative.
- **`c(flavor)` misreports the edition** as `IC` on StataSE, and `c(edition)` is
  unreliable. The generator derives the edition from the `c(MP)` / `c(SE)` 0/1 flags,
  with BE by elimination (`c(BE)` is undefined and `cond()` evaluates all branches
  eagerly, so it must not be referenced). `"SE"` is simply the committed golden's
  current value.
- **SSC has no version history.** `ssc install` always fetches latest and there is
  no lockfile / archive to pin against. The LPDiD RA arm is exempt (`teffects` is
  native, pinned by `version 19`); the SSC arms (ImputationDiD, ETWFE/CS, LWDiD,
  LPDiD non-absorbing) record their SSC package versions in `meta.ssc_versions` so
  drift is at least detectable — new SSC arms should do the same.
