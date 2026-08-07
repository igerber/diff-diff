---
date: 2026-08-06
updated: 2026-08-07
branch: feat/twfeweights-r-compat
plan: (none)
session-log: (none)
status: complete
---

# Checkpoint — R-compat port of twfeweights / ptetools / badcontrols

## Goal (one sentence)
Port the full public API of the R packages `twfeweights`, `ptetools`, and `badcontrols` into `diff_diff/`, with R/Python numeric parity on the implemented subset.

## Where I am
- `diff_diff/twfeweights.py`, `diff_diff/ptetools.py`, `diff_diff/badcontrols.py` all exist and are exported from `diff_diff/__init__.py`.
- Work is committed and clean (no uncommitted changes on `feat/twfeweights-r-compat`).
- Specialized test set: `DIFF_DIFF_BACKEND=python .venv/bin/pytest tests/test_*` passes 49.
- R parity verified for: `twfe_weights`, `ptetools did_attgt`, `badcontrols` continuous + binary imputation.
- Ruff / black / mypy all clean.
- Not yet done: `ggpte` / `ggpte_cont` plotting, `attgt_noif`, `covid_attgt`.

## Progress since checkpoint (2026-08-06, QTT step)
- **Full QTT/QoTT block ported** (`diff_diff/ptetools.py`): `PTEQTTResult` container +
  `pte_qtt`, `compute_pte` (R `compute.pte` `(g,t)` loop, time-major cells, influence scaled
  n/n1, universal-base-period zero cells), `qtt_pte_aggregations` (overall/dynamic/group
  quantile curves), `qott_pte_aggregations` (treatment-effect distributions), and
  `qtt_empirical_bootstrap` (unit-level block bootstrap + `se`/`lower_pw`/`upper_pw`/
  `lower_ub`/`upper_ub` bands). `block_boot_sample` now accepts an RNG.
- **Parity bug fixed:** `_type1_quantile` uses R's exact `j == floor(j)` integer check (was
  `np.isclose`) — single-cohort R panel now matches to max diff 0.0.
- **`_qtt_crit_val`** verified vs R `qtt_crit_val` to ~1e-13 (goldens 3.9813488035109388 /
  3.6799194197997362 for alpha .05/.10).
- **Documented deviation from R:** `_aligned_cell_returns` fixes R's latent `merge`-reorder
  misalignment (multi-cohort case misaligns in R); note added to `REGISTRY.md`.
- Exported `PTEQTTResult`, `block_boot_sample`, `compute_pte`, `pte_qtt`,
  `qott_pte_aggregations`, `qtt_empirical_bootstrap`, `qtt_pte_aggregations` from
  `diff_diff/__init__.py`; docs entry in `docs/api/ptetools.rst`; CHANGELOG bullet.
- New tests: `tests/test_ptetools_qtt.py` (9 tests: aggregation goldens on R fixtures,
  cell ordering, QoTT structure, bootstrap columns/reproducibility, crit-val, container).
  Full ptetools/dose/mboot/QTT/docs_ia suite: 84 passed, 2 skipped.
- QTT parity reference artifacts: `/tmp/qtt_panel1.csv` (40-unit single cohort), `/tmp/qtt_int.csv`
  (6-unit integer panel), `/tmp/qtt_panel.csv` (60-unit two-cohort), `/tmp/qtt_src.txt` etc.
  (R deparse dumps). R golden: overall QTT matches to 0.0 on the 40-unit panel.

## File pointers
- **`process_dose_gt` ported** (`diff_diff/ptetools.py`) as a faithful consumer of an R-style
  `gt_results` dict + `ptep` options dict (per-cell `att.d`/`acrt.d`/`att.overall`/`acrt.overall`/
  `bread`/`Xe`, zero-padded `inffunc`), returning a complete `DoseResult` (ATT(d)/ACRT(d) curves,
  per-dose SEs, pointwise-or-simultaneous crit values, overall ATT/ACRT + SEs + influence functions).
- **`bspline_basis` helper** reproduces `splines2::bSpline` / `splines2::dbs` EXACTLY (verified live
  against installed R; golden values pinned in `tests/test_ptetools_process_dose_gt.py`): clamped
  boundary knots, `intercept=False` drops the first basis column, derivative basis via the standard
  knot/coefficient transform.
- **`mboot_se_and_crit`** turns `mboot2` draws into R-style IQR bootstrap SEs + sup-t crit value
  (R `quantile(type=1)`).
- **`DoseResult` extended** to carry the full `dose_obj` surface (overall_acrt, se/crit/inffunc
  fields, `simultaneous`, `alp`, `biters`) while keeping `pte_dose_results` backward-compatible.
- Exported `process_dose_gt`, `bspline_basis`, `mboot_se_and_crit` from `diff_diff/__init__.py`;
  docs entry in `docs/api/ptetools.rst`; CHANGELOG Added bullet.
- New tests: `tests/test_ptetools_process_dose_gt.py` (7 tests: splines2 golden parity for level +
  derivative, knot validation, end-to-end point estimates + shapes, seed reproducibility, order and
  missing-field rejection). Full ptetools+R-parity suite: 31 passed.
- R parity for dose NOT runnable end-to-end: the per-cell dose estimators producing `att.d`/`bread`/
  `Xe` live outside this repo's R reference, and the R `pte_default` self-call is RNG-dependent. The
  basis helper parity is covered by the splines2 golden tests.
- Open follow-ups for this port: `n1_vec`/`keep_mat` require a zero-padded (R `compute.pte`
  convention) `inffunc`; off-support rows in the generic `pte()` influence surface are NaN-padded and
  get `nan_to_num`'d in the port — note if a future dose estimator feeds it.

## File pointers
- `diff_diff/ptetools.py:138` — `dose_obj` result container (process_dose_gt will fill it)
- `diff_diff/ptetools.py:550` — `pte()` main loop (QTT variants branch off here)
- `diff_diff/ptetools.py:434` — `did_attgt` (base estimator that process_dose_gt consumes)
- `diff_diff/ptetools.py:741` — `mboot2` (multiplier bootstrap used by process_dose_gt)
- `diff_diff/twfeweights.py` — twfe_weights + post-lasso block, all parity-tested
- `tests/test_r_parity_new_features.py` + `tests/r_parity_reference.R` — live R parity harness
- `../references-ptetools/R/process_dose_gt.R` — reference source for the next port
- `../references-ptetools/R/empirical_bootstrap.R` — contains qtt_empirical_bootstrap
- `../references-ptetools/R/ggpte.R` — plotting reference
- `../references-ptetools/R/pte.R` — compute.pte / covid_attgt
- `../references-ptetools/R/classes.R` — attgt_noif

## Recent decisions
- `implicit_twfe_weights` parity dropped: R fixest segfaults on the small fixture, so no byte-level R comparison is possible.
- `did_post_lasso` parity dropped: the R source is incomplete (contains a `browser()` debug path).
- `scikit-learn` is an optional extra (`[ml]`), keeping the core dependency light.
- Only newly-added-feature tests are run per user instruction; the full diff-diff suite is NOT run.
- Dose SEs/critical values always use the multiplier bootstrap (matches R; analytical SEs unsupported in R too).

## Open questions
- Q1: Should `ggpte`/`ggpte_cont` return a matplotlib figure/axes, or a data-frame + plotting helper? (R returns a ggplot object; matplotlib has no ggplot analog.) — **RESOLVED:** matplotlib `Axes` / Plotly `Figure` wrappers; REGISTRY deviation noted (commit `440e1c3b`).
- Q2: Does `covid_attgt` deserve parity (it's a data-centric example), or just a thin constructor? — **RESOLVED:** ported as a DR with dCDH panel score; covered by `test_covid_attgt_reuses_drdid_panel_score_*`.
- Q3: Whether to keep porting at all beyond the dose + QTT steps, since the remaining surface is large. — **RESOLVED:** ported dose, full QTT/QoTT, plotting wrappers, badcontrols parametric/ML cross-fit, and documented the dropped-parity twfeweights tails.

## Next 1–3 actions
1. Full-port scope is complete and committed on `feat/twfeweights-r-compat`.
2. Remaining follow-up resolved: the high-level `pte()` influence surface now zero-pads off-support units and scales by `(n/n1)`, matching R `compute.pte` and the Python `compute_pte` (commit `133410e5`).

## Resume prompt
> Resuming from checkpoint `quality_reports/checkpoints/2026-08-06_twfeweights-r-compat.md`. Read it, then continue with the commit of the QTT block (action 1) and decide Q3.
