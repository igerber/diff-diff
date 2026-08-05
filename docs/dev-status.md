# Development Status & Monitoring

Current-state notes, monitoring, and platform quirks — not a backlog. Shippable work
lives in [TODO.md](../TODO.md); blocked / parked work and decisions on the record in
[DEFERRED.md](../DEFERRED.md). This file is repo-internal (excluded from the RTD build).

## Large Module Files

Target: ideally < 1000 lines per module; modules ≥3000 lines are candidates for splitting,
2000-3000 are monitored, 1000-2000 are accepted as a cohesion / scope trade-off. Updated
2026-07-19.

| File | Lines | Action |
|------|-------|--------|
| `chaisemartin_dhaultfoeuille.py` | 8812 | Consider splitting (per-path / placebos / survey IF / aggregation) |
| `linalg.py` | 5424 | Consider splitting (vcov surfaces) only if cohesion preserved — unified backend; vcov / solver paths tightly coupled |
| `staggered.py` | 4992 | Consider splitting — grew through survey + aggregation features |
| `had.py` | 4748 | Consider splitting (continuous / mass-point / event-study / survey paths) |
| `had_pretests.py` | 4664 | Consider splitting (Stute / Yatchew / QUG / joint pretests) |
| `diagnostic_report.py` | 4135 | Consider splitting (per-method renderers + provenance) |
| `spillover.py` | 3655 | Consider splitting |
| `two_stage.py` | 2430 | Monitor — exited the splitting band when the M-022 aggregate() migration extracted the Stage-2/GMM engine into `two_stage_aggregation.py` |
| `power.py` | 3488 | Consider splitting (power analysis + MDE + sample size) |
| `utils.py` | 3483 | Consider splitting |
| `synthetic_control_results.py` | 3294 | Consider splitting |
| `honest_did.py` | 3068 | Consider splitting |
| `imputation.py` | 1491 | Acceptable — dropped below 2000 when the M-021 aggregate() migration extracted the Theorem-3 engine into `imputation_aggregation.py` |
| `imputation_aggregation.py` | 1858 | Acceptable — verbatim-moved Theorem-3 aggregation/variance engine (M-021/M-118) |
| `two_stage_aggregation.py` | 1555 | Acceptable — verbatim-moved Stage-2/GMM aggregation engine (M-022/M-119) |
| `synthetic_did.py` | 2826 | Monitor — variance methods + survey paths |
| `business_report.py` | 2728 | Monitor — per-method narrative renderers |
| `survey.py` | 2681 | Monitor — grew with Phase 6 features |
| `synthetic_control.py` | 2526 | Monitor |
| `prep_dgp.py` | 2524 | Monitor |
| `estimators.py` | 2441 | Monitor |
| `continuous_did.py` | 2459 | Monitor |
| `sun_abraham.py` | 2314 | Monitor |
| `triple_diff.py` | 2231 | Monitor |
| `wooldridge.py` | 2192 | Monitor |
| `practitioner.py` | 2113 | Monitor — grew with per-estimator handlers (was 1511 on 2026-07-13) |
| `efficient_did.py` | 1729 | Acceptable — dropped below 2000 when the M-023 aggregate() migration extracted the aggregation mixin into `efficient_did_aggregation.py` (~520 lines, below this table's floor) |
| `chaisemartin_dhaultfoeuille_results.py` | 2004 | Monitor |
| `results.py` | 1948 | Acceptable |
| `_rdrobust_port.py` | 1913 | Acceptable |
| `pretrends.py` | 1879 | Acceptable |
| `prep.py` | 1878 | Acceptable |
| `efficient_did_covariates.py` | 1818 | Acceptable |
| `staggered_triple_diff.py` | 1680 | Acceptable |
| `trop_local.py` | 1662 | Acceptable |
| `lpdid.py` | 1607 | Acceptable |
| `stacked_did.py` | 1589 | Acceptable |
| `changes_in_changes.py` | 1429 | Acceptable |
| `_nprobust_port.py` | 1425 | Acceptable |
| `bacon.py` | 1376 | Acceptable |
| `local_linear.py` | 1325 | Acceptable |
| `trop_global.py` | 1298 | Acceptable |
| `datasets.py` | 1224 | Acceptable |
| `rdd.py` | 1218 | Acceptable |
| `staggered_aggregation.py` | 1204 | Acceptable |
| `chaisemartin_dhaultfoeuille_bootstrap.py` | 1175 | Acceptable |
| `conley.py` | 1140 | Acceptable |
| `rdplot.py` | 1135 | Acceptable |
| `trop.py` | 1026 | Acceptable |
| `profile.py` | 1001 | Acceptable |

## Standard Error Consistency

`vcov_type` has subsumed the previously-proposed `se_type` knob. `DifferenceInDifferences`
and `TwoWayFixedEffects` accept `vcov_type ∈ {classical, hc1, hc2, hc2_bm, conley}`
(the validated set in `linalg.py::_VALID_VCOV_TYPES`); cluster-robust variance comes from
`cluster=` alongside the heteroscedasticity kind (`hc1+cluster` ⇒ CR1 Liang-Zeger;
`hc2_bm+cluster` ⇒ CR2 Bell-McCaffrey, including the weighted WLS-CR2 port; the N>1
absorbed-FE + weights composition is supported via iterative alternating-projection demeaning, #586);
wild cluster bootstrap is the separate `inference="wild_bootstrap"` path. Threading
`vcov_type` through the 8 standalone estimators is **complete** (Phase 1b); four
(`CallawaySantAnna`, `TripleDifference`, `ImputationDiD`, `EfficientDiD`) are permanently
narrow to `{hc1}` per their influence-function variance, and `TwoStageDiD` is likewise
narrow (Gardner GMM meat has no single cross-stage hat matrix). The per-estimator
`vcov_type="conley"` extensions: SunAbraham + WooldridgeDiD-OLS are **shipped**; the
IF/GMM estimators are tracked in
[DEFERRED.md → Paper-gated](../DEFERRED.md#paper-gated--needs-methodology-derivation).

## Type Annotations

`mypy diff_diff` is **enforced at zero errors** by the Lint CI workflow's Mypy job at the
pinned `mypy==2.3.0` (ungated, every PR push; pins synced with the `dev` extra via
`TestLintWorkflowPinSync`). `[tool.mypy] python_version` targets `"3.10"` (mypy >= 2.2
cannot target 3.9); the Python 3.9 library floor is covered by a dedicated runtime leg in
the gated test matrix instead. Zero is reached with documented suppressions — tightening them
is tracked in [TODO.md](../TODO.md) → Testing / docs:

- Global `disable_error_code = ["arg-type", "return-value", "var-annotated", "assignment"]`
  (numpy-stub compatibility, long-standing).
- Per-module override: `diff_diff.prep_dgp` disables `[index]` (seeded-DGP Optional
  covariate arrays; see the comment in `pyproject.toml`).
- `matplotlib.*` / `plotly.*` use `follow_imports = "skip"` so local and CI runs see
  identical `Any` surfaces regardless of what plotting stubs are installed.
- A handful of reasoned inline `# type: ignore[code]` comments, each with a why-comment;
  `warn_unused_ignores = true` prevents them from going stale.

Mixin cross-class attribute access uses `TYPE_CHECKING`-guarded attribute/method stubs in
the bootstrap mixin classes; keep stubs in sync with implementations (stub drift shows up
as `[misc]` unpack-arity errors).

## Test Coverage

Visualization tests skip when matplotlib / plotly are not installed (see
`pytest.importorskip` markers in `tests/test_visualization*.py`).

## RuntimeWarnings — Apple Silicon M4 BLAS bug (numpy < 2.3)

Spurious RuntimeWarnings ("divide by zero", "overflow", "invalid value") are emitted by
`np.matmul`/`@` on Apple Silicon M4 + macOS Sequoia with numpy < 2.3, for matrices with
≥260 rows. They **do not affect correctness** (coefficients/fitted values are valid, designs
full rank). Root cause: Apple's BLAS SME kernels corrupt the FP status register
([numpy#28687](https://github.com/numpy/numpy/issues/28687),
[#29820](https://github.com/numpy/numpy/issues/29820); fixed in numpy ≥ 2.3 via
[PR #29223](https://github.com/numpy/numpy/pull/29223)). Not reproducible on M3, Intel, or
Linux.

- `linalg.py:162` — warnings in fitted-value computation (`X @ coefficients`); seen in
  `test_prep.py` during treatment-effect recovery (n > 260).
- `triple_diff.py:307,323` — warnings in propensity-score computation (IPW/DR with
  covariates); logistic-regression overflow in edge cases (separate from the BLAS bug).
- **Long-term:** revert to the `@` operator when numpy ≥ 2.3 becomes the minimum supported
  version.
