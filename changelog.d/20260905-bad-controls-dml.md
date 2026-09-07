### Added
- **`DMLDiD` bad-control lane (Caetano, Callaway, Payne & Sant'Anna 2026, "Difference-in-differences with 'bad controls'", arXiv:2608.03881; PR-B of the bad-controls initiative).**
  `DMLDiD(...).fit(..., bad_control="x", bad_control_covariates=[...])` swaps the per-cell
  Chang (2020) score for the paper's Neyman-orthogonal doubly-robust score (Eq. 10 / 11,
  Algorithm 1; `ccps_panel_score` / `ccps_panel_score_augmented` in `_dr_scores.py`):
  parallel trends conditional on the bad control's UNTREATED path, with its untreated
  evolution identified by covariate unconfoundedness given its base-period value, the
  optional `W` covariates (`[outcome]` = the base-period outcome, the paper's Remark 5
  recommendation; default no `W`) and the covariates `Z`. Four cross-fitted nuisances
  per cell, including the two NESTED second stages (`nu`, `omega`): in-sample fold-k
  targets for the parametric `linear`/`logit` built-ins (the paper's Assumption-8
  plug-in), split-half swap-and-average for `ridge` / `sieve` / user learners
  (footnote 9); `omega` clipped to `[0, (1-trim)/trim]` with a warning. Every cell also
  reports the paper's Remark 6 pre-test `ATT_X(g,t)` (the effect of treatment on the bad
  control itself, an AIPW mean-effect diagnostic with its own analytical SE, sharing the
  ATT's cluster / df branch) via `results.bad_control_summary()` /
  `results.bad_control_diagnostics`, with `att_x` / `se_x` joined into
  `to_dataframe()` and new `summary()` header lines. Panel lane only, bare `cluster=`
  only (`survey_design=` raises), `anticipation=0` and `base_period="varying"` only;
  the bad control may not appear in `covariates`; `bad_control=None` is the untouched
  pre-existing code path (bit-identical). The headline `att` keeps the CS "simple"
  weighting (not the paper's Remark 4 overall; TODO row). Validation: score-level
  paired double robustness + four-direction Neyman orthogonality + reduction to
  `chang_panel_score` at 1e-14; a numpy oracle of Eq. 11 / Algorithm 1 at 1e-12
  (ATT, SE, ATT_X); oracle user learners on the split-half branch; the Supplementary
  Appendix's DGP 1 / DGP 4 recovery (`ATT = 1.00`, `ATT_X = 0.50`) with slow MC
  coverage; and tolerance-based black-box goldens against the authors' GPL-3 R package
  `badcontrols` 1.0.0 (executed only, never read; 10-seed means on both sides,
  `|Δatt| < 0.5 SE` per cell) in `tests/test_dml_did_bad_controls_parity.py` with the
  generator `benchmarks/R/generate_badcontrols_golden.R` and commit-pinned installer
  in `benchmarks/R/requirements.R`. `BusinessReport` / target-parameter /
  `practitioner_next_steps` carry the bad-control identification text, the second
  citation and the refit snippet arguments.

### Changed
- **`_crossfit.py` deep-copy fallback warning** now reads `"_crossfit: could not
  deep-copy ..."` (was `"cross_fit_predict: ..."`) and is attributed to the frame that
  advances the fold generator; the module gains the per-fold generator
  `iter_fold_fits` / `FoldFit` that `cross_fit_predict` is now a consumer of
  (behavior-preserving refactor; every existing pin unchanged).

### Documentation
- REGISTRY `DMLDiD` "Bad-control extension (CCPS 2026)" block (equations as
  implemented, nested-stage / omega-clip / W-default / base-period / anticipation /
  fail-closed / complete-case / aggregation-weight / ATT_X / validation-scope Notes),
  the CallawaySantAnna Approach-1 Note (a pre-treatment bad control in `covariates`
  computes Proposition 3 on the panel lane), the infrastructure section's fourth score
  family and `iter_fold_fits` contract; `docs/api/dml_did.rst` methodology sub-block,
  restrictions and a runnable snippet; `docs/api/staggered.rst` "Covariates and bad
  controls"; the paper review's requirements checklist flipped with library
  annotations; guides (`llms.txt` signature + Diagnostics entry, `llms-full.txt`,
  practitioner pitfall 3 rewrite, autonomous matrix); `choosing_estimator.rst`,
  `practitioner_decision_tree.rst`, `docs/index.rst`, README one-liners; survey
  roadmap / survey theory carve-outs; `docs/references.rst`; the regenerated
  variance-conventions table (`dml_did_bad_control` row); `benchmarks/R/README.md`.
