# R `conleyreg` parity benchmark for Conley spatial HAC SE

`benchmarks/R/generate_conley_golden.R` produces the golden JSON used by
`tests/test_conley_vcov.py::TestConleyParityR` to verify that diff-diff's
`vcov_type="conley"` matches R `conleyreg` (Düsterhöft 2021, CRAN v0.1.9)
to ≤ 1e-6 on three benchmark fixtures.

## Why R `conleyreg`

`conleyreg` is the canonical open-source Conley (1999) implementation in R
(Christian Düsterhöft, https://github.com/cdueben/conleyreg). It uses
RcppArmadillo for the inner loops and is widely cited in applied work.
Stata `acreg` (Colella et al. 2019) is the parallel canonical
implementation in the Stata ecosystem; we cite both in REGISTRY but only
parity-test against `conleyreg` because it is free and open source.

## Earth radius constant

`conleyreg::haversine_dist` uses **6371.01 km** (mean Earth radius) — see
[`src/distance_functions.cpp`](https://github.com/cdueben/conleyreg/blob/master/src/distance_functions.cpp).
diff-diff's `_CONLEY_EARTH_RADIUS_KM` is set to `6371.01` to match. WGS-84
equatorial radius is 6378.137 km; the 0.01 km vs 6371.0 delta is
methodologically negligible (Earth mean radius is approximate at many
more digits) but matters for the 1e-6 cross-language parity bound.

## Regenerating the fixtures

Requires:
- R installed (`/opt/homebrew/bin/Rscript` on Apple Silicon Mac)
- System libraries: `brew install gdal proj geos pkg-config udunits`
  (needed by sf, lwgeom — transitive deps of conleyreg)
- R packages: `Rscript -e 'install.packages(c("conleyreg","sf","lwgeom","jsonlite"))'`

```bash
cd benchmarks/R
Rscript generate_conley_golden.R
# Produces benchmarks/data/r_conleyreg_conley_golden.json
```

The output JSON is **committed to the repo** so CI doesn't need R. Only
re-run when:
- conleyreg is updated (verify version in `meta.tool` field)
- The set of benchmark fixtures changes

## Skip behavior

`tests/test_conley_vcov.py::TestConleyParityR` calls
`pytest.skip("Golden JSON not present...")` when the JSON is absent, so
CI passes without R. The 64 internal tests (`TestConleyKernels`,
`TestConleyDistanceMetrics`, `TestConleyReductions`,
`TestConleyDirectHelper`, `TestConleyValidatorHelpers`,
`TestConleyValidationDispatch`, `TestConleyEstimatorIntegration`,
`TestConleyTWFE`, `TestConleyEstimatorValidation`,
`TestConleySetParamsAtomicity`, `TestConleyLinearRegression`,
`TestConleyReductionsAddendum`) verify the implementation independently.

## Fixtures

Six fixtures total: three cross-sectional (Phase 1) and three panel
fixtures with `lag_cutoff > 0` (Phase 2, block-decomposed Conley).

**Cross-sectional** (`build_fixture`, `lag_cutoff=0`):

| Fixture | n | k | Cutoff | Stress test |
|---|---|---|---|---|
| `small_haversine` | 50 | 2 | 500 km | Small-n, simple regressor |
| `dense_haversine` | 200 | 3 | 1000 km | Dense, 2 covariates, large cutoff |
| `lat_lon_realistic` | 300 | 3 | 200 km | Continental US lat/lon range |

**Panel block-decomposed** (`build_panel_fixture`, `lag_cutoff > 0`):

| Fixture | n_units × T | k | Cutoff | Lag | Stress test |
|---|---|---|---|---|---|
| `panel_haversine_lag1` | 60 × 3 | 2 | 500 km | 1 | Short panel, 1-period serial |
| `panel_haversine_lag2` | 80 × 5 | 3 | 1000 km | 2 | Longer panel, 2-period serial |
| `panel_lat_lon_realistic_lag1` | 100 × 4 | 3 | 200 km | 1 | Continental US, 1-period serial |

Each unit's `(lat, lon)` is time-invariant within the panel fixtures; the
block-decomposed sandwich (within-period spatial + within-unit Bartlett
serial) is independent of `lag_cutoff` for within-period contributions
and matches R `conleyreg::time_dist.cpp` for the serial component.

The euclidean code path (`conley_metric="euclidean"`) is verified
internally against `scipy.spatial.distance.cdist` in
`tests/test_conley_vcov.py::TestConleyDistanceMetrics::test_pairwise_distance_euclidean_matches_pdist`.
conleyreg's planar code path requires an `sf` CRS specification, which
adds noise without methodological value for parity testing.

## JSON schema

```json
{
  "meta": {
    "generated_at": "2026-05-10",
    "earth_radius_km": 6371.01,
    "tool": "R conleyreg 0.1.9 (Düsterhöft 2021)"
  },
  "small_haversine": {
    "x": [<n*k floats, row-major>],
    "x_shape": [n, k],
    "y": [<n floats>],
    "coords": [<n*2 floats: lat, lon, row-major>],
    "coords_shape": [n, 2],
    "metric": "haversine",
    "cutoff_km": 500.0,
    "kernel": "bartlett",
    "vcov": [<k*k floats, row-major>],
    "vcov_shape": [k, k],
    "n": <int>,
    "k": <int>
  },
  "dense_haversine": { ... },
  "lat_lon_realistic": { ... }
}
```

The R script transposes matrices before `as.vector` flatten so that
NumPy's `np.asarray(...).reshape(shape)` (row-major / C-order) decodes
the same orientation R wrote. Without the transpose, R's column-major
flatten misaligns when reshaped row-major.

## Known constraints

- `conleyreg` requires `unit` and `time` columns even with `lag_cutoff=0`
  (cross-sectional). The script fakes them with `unit = 1:n, time = 1L`;
  conleyreg emits a `Number of time periods: 1. Treating data as
  cross-sectional` warning which is informational.
- `conleyreg` uses OpenMP for parallelism; on macOS Apple Silicon with
  R's default toolchain, the `OpenMP not detected` warning is normal —
  the package falls back to single-threaded mode without affecting
  numerical output.

---

# R `rdrobust` bandwidth-selection golden fixtures

`benchmarks/R/generate_rdrobust_golden.R` produces
`benchmarks/data/rdrobust_golden.json`, consumed by
`tests/test_rdrobust_port.py` to verify that
`diff_diff._rdrobust_port.rdbwselect` matches R `rdrobust::rdbwselect`
(Calonico, Cattaneo, Farrell & Titiunik) on SHARP bandwidth selection
across all 10 selectors at rtol ≤ 1e-9 (17 configs; this fixture predates
fuzzy support and is deliberately never regenerated). Fuzzy bandwidth and
estimation parity lives in `benchmarks/data/rdrobust_estimates_golden.json`
(generator `generate_rdrobust_estimates_golden.R`), pinned by
`tests/test_rdrobust_port.py::TestFuzzyPortGoldenParity` and
`tests/test_rdd_parity.py`.

## Version pin

The parity target is the **CRAN 4.0.0 release** (source tarball sha256
`78f0d6b4bdec4091cc8f42f6f1598704747f95926446d3aaee381ea1d613a36f`), the
version users install. Do NOT regenerate with the GitHub development tree
(4.1.0-dev): it changes nearest-neighbor tie handling (`nn_tol`), the
`stdvars` default, and the bwcheck floor, and its bandwidths differ from the
released package. The generator hard-fails unless
`packageVersion("rdrobust") == "4.0.0"`.

## Senate data provenance

`benchmarks/data/rdrobust_senate.csv` (56KB) is the canonical rdrobust
example dataset: U.S. Senate election vote shares and Democratic victory
margins, 1914-2010, from Cattaneo, Frandsen & Titiunik (2015, *Journal of
Causal Inference* 3(1), 1-24). It is distributed publicly by the rdrobust
authors with their software (https://rdpackages.github.io/rdrobust/) and is
vendored here as a real-data parity anchor: its 38 tied margin values
exercise the mass-points machinery, and `masspoints="off"` reproduces the
bandwidths printed in Calonico, Cattaneo, Farrell & Titiunik (2017, *Stata
Journal* 17(2), 372-404) exactly, anchoring the golden files against
published numbers independent of our own R invocation.

## Regenerating

The generator hard-requires exactly 4.0.0. If CRAN's current release has
moved on, install the pinned version from the archive and verify the
tarball hash first:

```sh
# Verify the source of record (must print the sha256 below):
curl -sfLO https://cran.r-project.org/src/contrib/Archive/rdrobust/rdrobust_4.0.0.tar.gz \
  || curl -sfLO https://cran.r-project.org/src/contrib/rdrobust_4.0.0.tar.gz
shasum -a 256 rdrobust_4.0.0.tar.gz
# expected: 78f0d6b4bdec4091cc8f42f6f1598704747f95926446d3aaee381ea1d613a36f

R CMD INSTALL rdrobust_4.0.0.tar.gz
Rscript benchmarks/R/generate_rdrobust_golden.R
```

# R `rddensity` manipulation-test golden fixtures

`benchmarks/R/generate_rddensity_golden.R` produces
`benchmarks/data/rddensity_golden.json`, consumed by
`tests/test_rddensity.py` to verify that `diff_diff.RDDensityTest` matches
R `rddensity::rddensity()` + `rddensity::rdbwdensity()` (Cattaneo, Jansson
& Ma) across 47 configs (five synthetic DGPs with embedded R-drawn samples,
plus the vendored `rdrobust_senate.csv` and `rddensity_headstart.csv` - the
latter fetched from the CJM 2020 replication repository
`rdpackages-replication/CJM_2020_JASA`, sha256
`28f42a04ca7392e786e5f93ba311cdc91489a293c061a3bc59f53f4cfc536ce9`).
The parity target is CRAN **rddensity 3.0** - pin the install to the
versioned tarball, never an unpinned `install.packages()`:

```sh
# Verify the source of record (must print the sha256 below):
curl -sfLO https://cran.r-project.org/src/contrib/rddensity_3.0.tar.gz \
  || curl -sfLO https://cran.r-project.org/src/contrib/Archive/rddensity/rddensity_3.0.tar.gz
shasum -a 256 rddensity_3.0.tar.gz
# expected: a9c45ab0f6b86ead4d91084db16513d4156b7f59b0472510b63deb5dee6f305d

R CMD INSTALL rddensity_3.0.tar.gz
Rscript benchmarks/R/generate_rddensity_golden.R
```

The generator hard-asserts `packageVersion("rddensity") == "3.0"`, embeds
every synthetic sample at 17 significant digits (R RNG streams are not
reproducible from numpy), and aborts loudly if the mass-point fixtures fail
to separate the two `nLocalMin`/`nUniqueMin` regularization gates (the
floor-gate configs exist to pin exactly that behavior).

# badcontrols (Caetano, Callaway, Payne & Sant'Anna 2026) black-box goldens

`benchmarks/data/badcontrols_golden.json` (+ the shared input panel
`benchmarks/data/badcontrols_panel.csv`) anchors the `DMLDiD` bad-control
lane (`fit(..., bad_control=, bad_control_covariates=)`) against the
authors' R package `badcontrols`. Consumed by
`tests/test_dml_did_bad_controls_parity.py`.

## Why this package, and the GPL black-box rule

`badcontrols` is the authors' reference implementation of the paper's
doubly-robust / DML estimator (arXiv:2608.03881). It is **GPL-3**; diff-diff
is MIT. The Python lane is derived from the paper alone
(`docs/methodology/papers/caetano-2026-review.md`), and the R package is
used ONLY as an executed oracle: the generator calls `didbc()` and records
its outputs. Contributors must never read `R/*.R` of that package while
working on `diff_diff/` (a direct port was rejected on licensing grounds).

## Regenerating

```sh
Rscript benchmarks/R/requirements.R              # installs badcontrols at the pinned commit + ptetools 1.0.0
Rscript benchmarks/R/generate_badcontrols_golden.R
pytest tests/test_dml_did_bad_controls_parity.py  # must NOT skip locally
```

The generator hard-fails unless `badcontrols` is 1.0.0 installed from
GitHub commit `651ccc925776125bb9233d76867c862107ea0ba5` (no tags or
releases exist and HEAD also reports 1.0.0, so the `RemoteSha` is the pin)
and `ptetools` is 1.0.0.

## What is recorded

`simulate_bad_controls(n = 1000, T_max = 4)` under `set.seed(20260905)`,
then four `didbc(est_method = "dr_ml", nuisance_method = "parametric",
xformula = ~Z, base_period = "varying", anticipation = 0, bstrap = FALSE,
overlap_threshold = 1, nfolds = 5)` runs: `bad_control_formula = ~X` with
`bad_control_cov_formula = ~W` (not-yet-treated, then never-treated), `~Y`
(Remark 5's lagged outcome, not-yet-treated), and a no-bad-control run
(characterization only - R's `dr_ml` path is not the plain Chang score).
`overlap_threshold = 1` disables the package's silent fallback to its
imputation estimator on high-propensity cells.

Each run is recorded as the per-cell mean over `n_seeds = 10` seeds
(`att_gt_mean`: `group, t, att, V_analytical, att_seed_sd, n, n_treated`)
plus every per-seed table (`att_gt_per_seed`). `V_analytical` is the
diagonal of `att_gt$V_analytical` (the analytical covariance; `att_gt$se`
is bootstrap-based even with `bstrap = FALSE`) and the parity SE is
`sqrt(V_analytical / n)`. `true_att_gt` carries the simulator's truth.

## Tolerance rationale and skip behavior

`didbc()` cross-fits with its own fold draw (`nfolds = 1` errors), so no
bit-exact target exists; single-seed fold noise is ~0.1-0.25 SE on most
cells and up to ~0.5 SE on the smallest ones. Averaging `n_seeds` seeds on
both sides shrinks the noise of the difference of means to ~0.1-0.2 SE, and
the test asserts `|mean_att_py - mean_att_R| < 0.5 SE_R` per cell plus an
SE ratio within 30% (runs 1-2 and run 3's post cells); run 3's pre cells
and run 4 are characterizations at 1.0 SE. A missing fixture skips the
module (isolated-install CI jobs copy `tests/` only).
