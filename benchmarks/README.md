# diff-diff Benchmarks

This directory contains benchmarks comparing diff-diff against equivalent R packages for validation and performance testing.

## Quick Start

```bash
# Generate synthetic data (required before first run)
python run_benchmarks.py --generate-data-only

# Run all benchmarks
python run_benchmarks.py --all

# Run specific estimator
python run_benchmarks.py --estimator callaway
```

Note: Synthetic data files and results are not committed to the repository.
Run `--generate-data-only` first to create the test datasets.

## Requirements

### Python
- diff-diff (this package)
- numpy, pandas, scipy

### R
Install R and required packages:

```bash
# Install R (macOS)
brew install r

# Install R packages
Rscript R/requirements.R
```

Required R packages:
- `did` - Callaway & Sant'Anna (2021)
- `synthdid` - Synthetic DiD (Arkhangelsky et al. 2021)
- `HonestDiD` - Rambachan & Roth (2023)
- `fixest` - Fast fixed effects estimation
- `jsonlite` - JSON interchange
- `data.table` - Fast data manipulation

### Stata

A few goldens are anchored against Stata where no runnable R reference exists.
Stata is node-locked single-user, so — like the R arm — the goldens are committed
and CI never needs Stata. Generators live in `stata/` and are run headless:

```bash
# macOS, StataSE 19 (binary is not on PATH by default)
STATA=/Applications/Stata/StataSE.app/Contents/MacOS/stata-se
$STATA -b do benchmarks/stata/requirements.do            # one-time SSC install
$STATA -b do benchmarks/stata/generate_lpdid_ra_golden.do
$STATA -b do benchmarks/stata/generate_lpdid_nonabsorbing_golden.do
$STATA -b do benchmarks/stata/generate_imputation_loo_golden.do
$STATA -b do benchmarks/stata/generate_etwfe_cs_golden.do
$STATA -b do benchmarks/stata/generate_reghdfe_kref_golden.do
$STATA -b do benchmarks/stata/generate_lwdid_golden.do   # see stata/README.md for its warm-up step
```

The `LPDiD` RA arm uses only **native** Stata commands (`teffects`), pinned by
`version 19`. The `ImputationDiD` arm depends on SSC packages
(`did_imputation`/`reghdfe`/`ftools`/`require`), the ETWFE/CS arm on
`drdid`/`csdid`/`jwdid`/`hdfe`, the reghdfe K_reference arm on `reghdfe`,
the LWDiD arm on the authors' `lwdid`, and the LPDiD non-absorbing arm on the
authors' `lpdid` (+ `boottest`/`egenmore`/`listreg`);
`version 19` does NOT pin SSC packages
(SSC has no version history) — install them once via `requirements.do` (the
generators do not auto-install) and each golden records version/checksum
metadata for drift detection. See `stata/README.md`.

## Directory Structure

```
benchmarks/
├── README.md                 # This file
├── run_benchmarks.py         # Main benchmark orchestrator
├── compare_results.py        # Result comparison utilities
├── R/
│   ├── requirements.R        # R package installation
│   ├── benchmark_did.R       # Callaway-Sant'Anna
│   ├── benchmark_synthdid.R  # Synthetic DiD
│   ├── benchmark_honest.R    # HonestDiD
│   └── benchmark_fixest.R    # Basic DiD / TWFE
├── stata/
│   ├── README.md                          # Stata arm docs
│   ├── requirements.do                    # one-time SSC install (did_imputation etc.)
│   ├── generate_lpdid_ra_golden.do        # LPDiD RA SE vs teffects ra
│   ├── generate_lpdid_nonabsorbing_golden.do  # LPDiD non-absorbing SEs vs the authors' lpdid
│   ├── generate_imputation_loo_golden.do  # ImputationDiD LOO SE vs did_imputation leaveout
│   ├── generate_etwfe_cs_golden.do        # ETWFE/CS vs jwdid + csdid (+ subsample ladder)
│   ├── generate_reghdfe_kref_golden.do    # clustered CR1 K_reference vs reghdfe (disconnected panel)
│   └── generate_lwdid_golden.do           # LWDiD vs the authors' lwdid (small-N, RI, event-study bootstrap)
├── python/
│   ├── utils.py              # Common utilities
│   ├── benchmark_callaway.py # CallawaySantAnna
│   ├── benchmark_synthdid.py # SyntheticDiD
│   ├── benchmark_honest.py   # HonestDiD
│   └── benchmark_basic.py    # Basic DiD / TWFE
├── data/
│   ├── synthetic/            # Generated test data
│   └── real/                 # Public datasets
└── results/
    ├── accuracy/             # Numerical comparison results
    └── performance/          # Timing results
```

## Estimator Comparisons

## DiD OVB sensitivity parity

`R/generate_did_ovb_parity.R` is an independent R oracle for the canonical
two-period omitted-variable-bias sensitivity implementation. It uses the
2006--2007 treated/never-treated slice of `data/real/mpdta.csv`, fixed folds,
and the paper's displayed plug-in and influence-function formulas. Regenerate
the JSON fixture with:

```bash
Rscript benchmarks/R/generate_did_ovb_parity.R \
  benchmarks/data/did_ovb_r_results.json \
  benchmarks/data/real/mpdta.csv
```

The corresponding Python test is `tests/test_did_ovb_r_parity.py`; it compares
the short ATT, scale components, bounds, RV, and XRV. This is a clean-room
parity harness and does not import the GPL-3 `dml.sensemakr` source.

The paper's minimum-wage application uses a separate external data file from
the authors' `CS_RR` repository. After exporting `data/min_wage_CS.rds` to CSV
in R, run the learner-level application check with:

```bash
Rscript -e 'write.csv(readRDS("data/min_wage_CS.rds"), "min_wage_CS.csv", row.names=FALSE)'
python benchmarks/python/benchmark_did_ovb_minwage.py min_wage_CS.csv
```

This runner uses sklearn random forests as an explicit approximation to the
paper's tuned `ranger` specification. Its output is a diagnostic, not a paper
replication claim; exact application parity requires matching `ranger`, its
cross-validation choices, fold assignments, and multiplier-bootstrap settings.

An R oracle using the actual `ranger` implementation is also provided. It
writes the fold assignment used by the R run so the Python estimator can use
the same units and folds:

```bash
R_LIBS_USER=/path/to/r-library Rscript benchmarks/R/benchmark_did_ovb_minwage.R \
  /path/to/min_wage_CS.rds /tmp/did_ovb_minwage_r.json \
  /tmp/did_ovb_minwage_folds.csv 42 5 1000 2 10 variance
PYTHONPATH=. python benchmarks/python/benchmark_did_ovb_minwage.py \
  /path/to/min_wage_CS.csv --folds-csv /tmp/did_ovb_minwage_folds.csv \
  --n-estimators 1000 --max-features 2 --min-samples-leaf 10 \
  --seed 42 --n-folds 5 --output /tmp/did_ovb_minwage_python.json
```

The R and Python forests are deliberately reported as separate estimates:
`ranger` and sklearn do not implement identical tree-growth and probability
prediction rules. The shared fold file isolates that learner implementation
difference from sample construction and cross-fitting differences. The
published application estimate is approximately `-0.0366`; the Python
diagnostic with the fixed seed and default learner settings is approximately
`-0.0362` on the shared data.

The Appendix E.1 simulation DGP has a shared R/Python fixture runner:

```bash
Rscript benchmarks/R/generate_did_ovb_simulation.R \
  benchmarks/data/did_ovb_simulation.csv \
  benchmarks/data/did_ovb_simulation_r.json
PYTHONPATH=. python benchmarks/python/benchmark_did_ovb_simulation.py \
  benchmarks/data/did_ovb_simulation.csv \
  --r-json benchmarks/data/did_ovb_simulation_r.json
```

This is the first single-draw cross-language check. The full 5,000-repetition
coverage and sensitivity-statistics Monte Carlo tables remain a separate task.

The compact Monte Carlo diagnostics are available as:

```bash
Rscript benchmarks/R/run_did_ovb_simulation_mc.R /tmp/ovb_mc_r.json 500 20
PYTHONPATH=. python benchmarks/python/run_did_ovb_simulation_mc.py \
  --n 500 --reps 20 --output /tmp/ovb_mc_python.json
```

Both runners report the mean short ATT, Monte Carlo standard deviation, mean
estimated standard error, bias relative to the true short ATT, and 95% CI
coverage. The paper-scale 5,000-repetition tables and sensitivity-statistic
coverage surfaces are not silently substituted by this compact diagnostic.

For the Appendix E.2 misspecification design, pass `--misspecified` to Python
and `1` as the seventh R argument. Both then fit the nuisance models using
`X*=exp(X/2)` while generating outcomes from the original `X`.

The Python-only forest diagnostic is
`benchmarks/python/run_did_ovb_simulation_forest.py`. It is not an R parity
test because the current R environment does not have `ranger` installed.

| diff-diff | Reference Package | Reference | Status |
|-----------|-----------|-----------|--------|
| `CallawaySantAnna` | `did::att_gt` | Callaway & Sant'Anna (2021) | ✓ Integrated |
| `SyntheticDiD` | `synthdid::synthdid_estimate` | Arkhangelsky et al. (2021) | ✓ Integrated |
| `DifferenceInDifferences` | `fixest::feols` | Standard DiD | ✓ Integrated |
| `LPDiD` (RA SE) | Stata `teffects ra ... atet` | Dube, Girardi, Jorda & Taylor (2025) | ✓ Integrated |
| `LPDiD` (non-absorbing SEs) | Stata `lpdid, nonabsorbing(...)` (authors' package) | Dube, Girardi, Jorda & Taylor (2025) §4.2 | ✓ Integrated |
| `ImputationDiD` (LOO SE) | Stata `did_imputation, leaveout` | Borusyak, Jaravel & Spiess (2024) App. A.9 | ✓ Integrated |
| `WooldridgeDiD` / `CallawaySantAnna` | Stata `jwdid` / `csdid` (+ G≈20..500 SE ladder) | Wooldridge (2025) / Callaway & Sant'Anna (2021) | ✓ Integrated |
| Clustered CR1 `K_reference` | Stata `reghdfe` + R `fixest` (disconnected-panel arms) | reghdfe/fixest ssc conventions | ✓ Integrated |
| `HonestDiD` | `HonestDiD::createSensitivityResults` | Rambachan & Roth (2023) | Planned |

Note: HonestDiD benchmark scripts exist but are not yet integrated into the main runner.

## Validation Criteria

### Accuracy
- **ATT difference**: < 1e-4 (absolute) or < 1% (relative)
- **SE difference**: < 10% (relative)
- **CI overlap**: Confidence intervals must contain each other's point estimates

### Performance
- Wall clock time (seconds)
- Memory usage (MB) - optional
- Scaling behavior

## Output Format

Results are saved as JSON files:

```json
{
  "estimator": "diff_diff.CallawaySantAnna",
  "overall_att": 2.0123,
  "overall_se": 0.1234,
  "timing": {
    "estimation_seconds": 0.456,
    "total_seconds": 0.789
  },
  "metadata": {
    "n_units": 200,
    "n_periods": 8,
    "n_obs": 1600
  }
}
```

## Adding New Benchmarks

1. Create R script in `R/benchmark_<name>.R`
2. Create Python script in `python/benchmark_<name>.py`
3. Add to `run_benchmarks.py`
4. Update documentation

## Reproducing Published Results

See `docs/benchmarks.rst` for full methodology and results.
