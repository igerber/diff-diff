# Project Memory

Persistent context for Claude Code sessions on diff-diff.

## Numerical Tolerances vs R

When comparing against R packages, expect these tolerance levels:

| Estimator | R Package | ATT Tolerance | SE Tolerance | Notes |
|-----------|-----------|---------------|--------------|-------|
| CallawaySantAnna | `did` | 1e-6 | 1e-4 | SE differences due to bootstrap randomness |
| SyntheticDiD | `synthdid` | 1e-6 | 1e-4 | Weight optimization may converge differently |
| HonestDiD | `HonestDiD` | 1e-4 | 1e-3 | LP solver differences (scipy vs R) |
| TWFE | `fixest` | 1e-10 | 1e-8 | Should match very closely |
| BasicDiD | `lm` | 1e-10 | 1e-8 | Should match very closely |

## Tutorial Fragility Notes

Notebooks ordered by execution time and fragility:

| Notebook | Typical Runtime | Fragility | Notes |
|----------|-----------------|-----------|-------|
| 01_basic_did | ~5s | Low | Simple, stable |
| 02_staggered_did | ~30s | Medium | Bootstrap can timeout on slow machines |
| 03_synthetic_did | ~10s | Low | Weight optimization is deterministic |
| 04_parallel_trends | ~5s | Low | Statistical tests, stable |
| 05_honest_did | ~15s | Medium | LP solver sensitive to numerical precision |
| 06_power_analysis | ~20s | Medium | Simulation-based, set seeds |
| 07_pretrends_power | ~10s | Low | Analytical, stable |
| 08_triple_diff | ~5s | Low | Simple estimation |
| 09_real_world_examples | ~30s | High | Downloads external data, network-dependent |
| 10_trop | ~45s | Medium | Factor model + bootstrap, computationally intensive |

### Known Fragile Cells

- **09_real_world_examples.ipynb**: Cells that download datasets may fail if servers are down
- **02_staggered_did.ipynb**: Bootstrap cells with n_boot=1000 may be slow
- **10_trop.ipynb**: Cross-validation for rank selection can be slow

## Performance Regression Thresholds

Alert if performance degrades beyond these thresholds (measured on 10K observations):

| Estimator | Max Acceptable Time | Current Baseline |
|-----------|---------------------|------------------|
| BasicDiD | 50ms | ~11ms |
| TWFE | 50ms | ~11ms |
| CallawaySantAnna (no bootstrap) | 200ms | ~50ms |
| CallawaySantAnna (500 bootstrap) | 2s | ~500ms |
| SyntheticDiD | 500ms | ~100ms |

## Common Debugging Patterns

### Bootstrap Tests Failing
1. Check if random seed is set: `np.random.seed(42)`
2. Increase tolerance for SE comparisons
3. Try increasing `n_boot` if variance is high

### R Comparison Discrepancies
1. Verify same data is being used (check row counts, column names)
2. Check if R package uses different default options
3. Look for edge cases in treatment timing definitions

### Numerical Instability
1. Check for near-singular matrices in OLS (condition number)
2. Verify no all-zero columns in design matrix
3. For SyntheticDiD: check if simplex projection is converging

### Memory Issues
1. CallawaySantAnna with many cohorts creates large bootstrap matrices
2. Consider `n_boot=None` for point estimates during debugging
3. TROP with high rank can use significant memory

## API Conventions

### Estimator Pattern
All estimators follow sklearn-like API:
```python
estimator = Estimator(params)
results = estimator.fit(df, outcome='y', treatment='treated', ...)
print(results.summary())
```

### Results Objects — CURRENT (3.x)
The canonical inference quintet is `att` / `se` / `t_stat` / `p_value` /
`conf_int` (never `.pvalue` / `.ci`). Every results class exposes all five,
but native STORAGE varies in 3.x (`overall_att` on the staggered family,
`avg_att` on MultiPeriodDiD, with canonical names as properties). `summary()`
everywhere; `to_dict()` / `to_dataframe()` on most classes. See
`.claude/../docs/methodology/REGISTRY.md` per estimator.

### Results Objects — 4.0 TARGET (do not write against this pre-4.0)
Target contract: `docs/v4-design.md` section 5; per-surface lifecycle:
`docs/v4-deprecations.yaml` (CI-enforced by `tests/test_v4_matrix.py`).

### Column Naming — CURRENT (3.x)
- `unit` unit id (`unit_col` on HAD; `group` on dCDH — both slated for 4.0)
- `time` calendar period, EXCEPT DifferenceInDifferences / TripleDifference /
  static TwoWayFixedEffects, where `time` is the 0/1 post dummy (TWFE warns
  on >2 unique values; 4.0 renames these to `post`, and TWFE's `time`
  becomes the event-study calendar column)
- `treatment` 0/1 treated-group indicator; `first_treat` cohort column
  (`cohort` on WooldridgeDiD — slated for 4.0)
- `covariates` covariate list (`controls` on dCDH — slated for 4.0)

### Column Naming — 4.0 TARGET (do not write against this pre-4.0)
Target vocabulary and rules: `docs/v4-design.md` section 8.

## Session Notes

<!-- Add notes from debugging sessions here -->

