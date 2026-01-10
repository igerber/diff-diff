# diff-diff Benchmarks

This directory contains benchmarks comparing diff-diff against equivalent R packages for validation and performance testing.

## Quick Start

```bash
# Run all benchmarks
python run_benchmarks.py --all

# Run specific estimator
python run_benchmarks.py --estimator callaway

# Generate synthetic data only
python run_benchmarks.py --generate-data-only
```

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

| diff-diff | R Package | Reference |
|-----------|-----------|-----------|
| `CallawaySantAnna` | `did::att_gt` | Callaway & Sant'Anna (2021) |
| `SyntheticDiD` | `synthdid::synthdid_estimate` | Arkhangelsky et al. (2021) |
| `HonestDiD` | `HonestDiD::createSensitivityResults` | Rambachan & Roth (2023) |
| `DifferenceInDifferences` | `fixest::feols` | Standard DiD |
| `TwoWayFixedEffects` | `fixest::feols` | Two-way FE |

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
