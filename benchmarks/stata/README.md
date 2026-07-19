# Stata parity benchmarks

Stata golden generators live here. They are the **first Stata arm** in the repo;
the pattern mirrors `benchmarks/R/` (a `generate_*` script writes a committed
golden JSON that a skip-guarded `tests/test_*_parity.py` reads, so CI never needs
Stata). Stata is node-locked single-user, so — exactly like the R arm — goldens
are committed and only regenerated locally.

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

## Known constraints

- **Batch mode always exits 0**, even on a hard error (`r(NNN);`). Never trust the
  shell exit code — parse the `.log` for `^r\([0-9]+\);` (the generator also runs
  an in-`.do` point smoke gate at 1e-8 that surfaces as `r(9);` on a gross port
  bug; the Python test's 1e-10 gate is authoritative).
- **`c(flavor)` misreports the edition** as `IC` on StataSE, and `c(edition)` is
  unreliable. The generator derives the edition from the `c(MP)` / `c(SE)` 0/1 flags,
  with BE by elimination (`c(BE)` is undefined and `cond()` evaluates all branches
  eagerly, so it must not be referenced). `"SE"` is simply the committed golden's
  current value.
- **SSC has no version history.** `ssc install` always fetches latest and there is
  no lockfile / archive to pin against. This arm is exempt because `teffects` is
  native (pinned by `version 19`), but any *future* SSC-dependent Stata generator
  must record `which <pkg>` output verbatim into its golden's `meta` so drift is
  at least detectable.
