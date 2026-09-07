### Documentation
- **Tutorial 33, "Bad Controls - Covariates That Treatment Can Affect"**
  (`docs/tutorials/33_bad_controls.ipynb`; PR-C of the Caetano, Callaway, Payne &
  Sant'Anna 2026 bad-controls initiative). On a staggered version of the paper's DGP 1
  it shows the naive TWFE regression with the bad control at `t` missing by the full
  treatment effect on the covariate, Approach 1 through base-period covariates with and
  without the confounders `W`, the `DMLDiD` bad-control lane and the choice of `W`
  (Remark 5's lagged outcome), how to read `bad_control_summary()` (pre-period rows
  pre-test MP-5/MP-8 and should be zero; post-period rows check that treatment moves the
  covariate), the event study, a ridge refit through the split-half nested stage, and
  the lane's restrictions. Registered in the tutorials index, `docs/tutorials/README.md`,
  `diff_diff/guides/llms.txt` (together with a line for tutorial 32) and
  `docs/doc-deps.yaml`; pinned by `tests/test_t33_bad_controls_drift.py` (code-cell
  hashes, quoted numbers, DGP re-derivation, pre-period narrative guard).

### Changed
- **`practitioner_next_steps()` names the bad-control lane**: on a `DMLDiD` fit with
  `bad_control` set the guidance banner reads "DMLDiD (CCPS 2026 bad-control score)"
  instead of the Chang (2020) label, matching the results `summary()` banner.
