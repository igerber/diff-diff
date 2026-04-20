# diff-diff Tutorials

This directory contains Jupyter notebook tutorials demonstrating the features of the `diff-diff` library.

## Notebooks

### 1. Basic DiD (`01_basic_did.ipynb`)
Introduction to Difference-in-Differences with `diff-diff`:
- Basic 2x2 DiD estimation
- Column-name and formula interfaces
- Adding covariates
- Fixed effects (dummy and absorbed)
- Two-Way Fixed Effects (TWFE)
- Cluster-robust standard errors
- Wild cluster bootstrap

### 2. Staggered DiD (`02_staggered_did.ipynb`)
Handling staggered treatment adoption with the Callaway-Sant'Anna estimator:
- Understanding staggered adoption
- Problems with TWFE in staggered settings
- **Goodman-Bacon decomposition**: Diagnosing *why* TWFE fails
- Group-time effects ATT(g,t)
- Aggregation methods (simple, group, event-study)
- Control group specifications
- Visualization

### 3. Synthetic DiD (`03_synthetic_did.ipynb`)
Synthetic Difference-in-Differences for few treated units:
- When to use Synthetic DiD
- Understanding unit and time weights
- Pre-treatment fit diagnostics
- Inference methods (bootstrap, placebo)
- Regularization tuning
- Comparison with standard DiD

### 4. Parallel Trends (`04_parallel_trends.ipynb`)
Testing assumptions and diagnostics:
- Visual inspection of trends
- Simple parallel trends tests
- Robust Wasserstein-based tests
- Equivalence testing (TOST)
- Placebo tests (timing, group, permutation)
- Event study as a diagnostic
- What to do if parallel trends fails

### 15. Efficient DiD (`15_efficient_did.ipynb`)
Efficient Difference-in-Differences (Chen, Sant'Anna & Xie 2025):
- Optimal weighting across comparison groups and baselines
- PT-All vs PT-Post assumptions
- Efficiency gains vs Callaway-Sant'Anna
- Event study and group-level aggregation
- Bootstrap inference and diagnostics

### 16. LPDiD (`16_lpdid.ipynb`)
Local Projections Difference-in-Differences for absorbing treatments:
- Event-study and pooled pre/post effects
- `rw` and `nocomp` option paths
- Official Stata `lpdid` example workflow
- Manual event-study plotting from `LPDiDResults`

## Running the Notebooks

1. Install diff-diff with dependencies:
```bash
pip install diff-diff
pip install matplotlib  # for visualizations
pip install jupyter     # to run notebooks
```

2. Start Jupyter:
```bash
jupyter notebook
```

3. Open any notebook and run the cells.

## Requirements

- Python 3.8+
- diff-diff
- numpy
- pandas
- matplotlib (optional, for visualizations)
