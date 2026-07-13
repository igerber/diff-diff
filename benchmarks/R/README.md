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
