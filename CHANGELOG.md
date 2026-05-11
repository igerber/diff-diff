# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Tutorial 21: HAD Pre-test Workflow** (`docs/tutorials/21_had_pretest_workflow.ipynb`) — composite pre-test walkthrough for `HeterogeneousAdoptionDiD` building on Tutorial 20's brand-campaign framing. Uses a 60-DMA × 8-week panel close in shape to T20's but with the dose distribution drawn from `Uniform[$0.01K, $50K]` (vs T20's `[$5K, $50K]`); the true support is strictly positive but very near zero, chosen so the QUG step in `did_had_pretest_workflow` fails-to-reject `H0: d_lower = 0` in this finite sample and the verdict text fires the load-bearing "Assumption 7 deferred" pivot for the upgrade-arc narrative. (HAD's `design="auto"` selector — a separate min/median heuristic at `had.py::_detect_design`, NOT the QUG p-value — independently lands on the `continuous_at_zero` identification path with target `WAS` on this panel because `d.min() < 0.01 * median(|d|)`. The QUG test and the design selector are independent rules that point to the same identification path here.) Walks through three surfaces: (a) `did_had_pretest_workflow(aggregate="overall")` on a two-period collapse, where the verdict explicitly flags Step 2 (Assumption 7 pre-trends) as not run because a single pre-period structurally cannot support a pre-trends test, and the structural fields `pretrends_joint` / `homogeneity_joint` are both `None`; (b) `did_had_pretest_workflow(aggregate="event_study")` on the full multi-period panel, where the verdict reads "TWFE admissible under Section 4 assumptions" because all three testable diagnostics (QUG + joint pre-trends Stute over 3 horizons + joint homogeneity Stute over 4 horizons) fail-to-reject — non-rejection evidence under finite-sample power and test specification, not proof that the identifying assumptions hold; and (c) a side panel exercising both `yatchew_hr_test` null modes — `null="linearity"` (default, paper Theorem 7) vs `null="mean_independence"` (Phase 4 R-parity with R `YatchewTest::yatchew_test(order=0)`) — on the within-pre-period first-difference paired with post-period dose, illustrating the stricter null's larger residual variance (`sigma2_lin` 7.01 vs 6.53) and smaller p-value (0.29 vs 0.49). Companion drift-test file `tests/test_t21_had_pretest_workflow_drift.py` (16 tests pinning panel composition, both verdict pivots, structural anchors on both paths, deterministic QUG / Yatchew statistics, bootstrap p-value tolerance bands per `feedback_bootstrap_drift_tests_need_backend_tolerance`, and `HAD(design="auto")` resolution to `continuous_at_zero` on this panel). T20's "Composite pretest workflow" Extensions bullet updated with a forward-pointer to T21. T22 weighted/survey HAD tutorial remains queued as a separate notebook PR.
- **`ChaisemartinDHaultfoeuille.by_path` and `paths_of_interest` now compose with `survey_design`** for analytical Binder TSL SE and replicate-weight bootstrap variance. The `NotImplementedError` gate at `chaisemartin_dhaultfoeuille.py:1233-1239` is replaced by a per-path multiplier-bootstrap-only gate (`survey_design + n_bootstrap > 0` under by_path / paths_of_interest still raises, since the survey-aware perturbation pivot for path-restricted IFs is methodologically underived). Per-path SE routes through the existing `_survey_se_from_group_if` cell-period allocator: the per-period IF (`U_pp_l_path`) is built with non-path switcher-side contributions skipped (control contributions are unchanged, matching the joiners/leavers IF convention; preserves the row-sum identity `U_pp.sum(axis=1) == U`), cohort-recentered via `_cohort_recenter_per_period`, then expanded to observations as `psi_i = U_pp[g_i, t_i] · (w_i / W_{g_i, t_i})`. Replicate-weight designs unconditionally use the cell allocator (Class A contract from PR #323). New `_refresh_path_inference` helper post-call refreshes `safe_inference` on every populated entry across `multi_horizon_inference`, `placebo_horizon_inference`, `path_effects`, and `path_placebos` so all four surfaces use the same final `df_survey` after per-path replicate fits append `n_valid` to the shared accumulator. Path-enumeration ranking under `survey_design` remains unweighted (group-cardinality, not population-weight mass). Lonely-PSU policy stays sample-wide, not per-path. Telescope invariant: on a single-path panel, per-path SE matches the global non-by_path survey SE bit-exactly. **No R parity** — R `did_multiplegt_dyn` does not support survey weighting; this is a Python-only methodology extension. The global non-by_path TSL multiplier-bootstrap path is unaffected (anti-regression test `tests/test_chaisemartin_dhaultfoeuille.py::TestByPathSurveyDesignAnalytical::test_global_survey_plus_n_bootstrap_still_works` locks the per-path-only scope of the new gate). Cross-surface invariants regression-tested at `TestByPathSurveyDesignAnalytical` (~17 tests across gate / dispatch / analytical SE / replicate-weight SE / per-path placebos / `trends_linear` composition / unobserved-path warnings / final-df refresh regressions) and `TestByPathSurveyDesignTelescope`. See `docs/methodology/REGISTRY.md` §`ChaisemartinDHaultfoeuille` `Note (Phase 3 by_path ...)` → "Per-path survey-design SE" for the full contract.
- **Inference-field aliases on staggered result classes** for adapter / external-consumer compatibility. Read-only `@property` aliases expose the flat `att` / `se` / `conf_int` / `p_value` / `t_stat` names (matching `DiDResults` / `TROPResults` / `SyntheticDiDResults` / `HeterogeneousAdoptionDiDResults`) on every result class that previously only carried prefixed canonical fields: `CallawaySantAnnaResults`, `StackedDiDResults`, `EfficientDiDResults`, `ChaisemartinDHaultfoeuilleResults`, `StaggeredTripleDiffResults`, `WooldridgeDiDResults`, `SunAbrahamResults`, `ImputationDiDResults`, `TwoStageDiDResults` (mapping to `overall_*`); `ContinuousDiDResults` (mapping to `overall_att_*`, ATT-side as the headline, ACRT-side accessible unchanged via `overall_acrt_*`); `MultiPeriodDiDResults` (mapping to `avg_*`). `ContinuousDiDResults` additionally exposes `overall_se` / `overall_conf_int` / `overall_p_value` / `overall_t_stat` aliases for naming consistency with the rest of the staggered family. Aliases are pure read-throughs over the canonical fields — no recomputation, no behavior change — so the `safe_inference()` joint-NaN contract (per CLAUDE.md "Inference computation") is inherited automatically (NaN canonical → NaN alias, locked at `tests/test_result_aliases.py::test_pattern_b_aliases_propagate_nan`). The native `overall_*` / `overall_att_*` / `avg_*` fields remain canonical for documentation and computation. Motivated by the `balance.interop.diff_diff.as_balance_diagnostic()` adapter (`facebookresearch/balance` PR #465) which calls `getattr(res, "se", None)` / `getattr(res, "conf_int", None)` without a fallback chain — pre-alias, every staggered result class returned `None` on those keys, silently dropping `se` and `conf_int` from the adapter's diagnostic dict. 23 alias-mechanic + balance-adapter regression tests at `tests/test_result_aliases.py`. Patch-level (additive on stable surfaces).
- **`ChaisemartinDHaultfoeuille.by_path` + non-binary integer treatment** — `by_path=k` now accepts integer-coded discrete treatment (D in Z, e.g. ordinal `{0, 1, 2}`); path tuples become integer-state tuples like `(0, 2, 2, 2)`. The previous `NotImplementedError` gate at `chaisemartin_dhaultfoeuille.py:1870` is replaced by a `ValueError` for continuous D (e.g. `D=1.5`) at fit-time per the no-silent-failures contract — the existing `int(round(float(v)))` cast in `_enumerate_treatment_paths` is now defensive (no-op for integer-coded D). Validated against R `did_multiplegt_dyn(..., by_path)` for D in `{0, 1, 2}` via the new `multi_path_reversible_by_path_non_binary` golden-value scenario (78 switchers, 3 paths, single-baseline custom DGP, F_g >= 4): per-path point estimates match R bit-exactly (rtol ~1e-9 on event horizons; rtol+atol envelope for placebo near-zero values), per-path SE inherits the documented cross-path cohort-sharing deviation (~5% rtol observed; SE_RTOL=0.15 envelope). **Deviation from R for D >= 10:** R's `did_multiplegt_by_path` derives the per-path baseline via `path_index$baseline_XX <- substr(path_index$path, 1, 1)`, which captures only the first character of the comma-separated path string (e.g. for `path = "12,12,..."` it captures `"1"` instead of `"12"`); this mis-allocates R's per-path control-pool subset for D >= 10. Python's tuple-key matching is correct in this regime — the per-path point estimates we compute are correct; R's per-path subset for the same path is buggy. The shipped parity scenario stays in `D in {0, 1, 2}` to avoid the R bug. R-parity test at `tests/test_chaisemartin_dhaultfoeuille_parity.py::TestDCDHDynRParityByPathNonBinary`; cross-surface invariants regression-tested at `tests/test_chaisemartin_dhaultfoeuille.py::TestByPathNonBinary`.
- **New `paths_of_interest` kwarg on `ChaisemartinDHaultfoeuille`** for user-specified treatment-path subsets, alternative to `by_path=k`'s top-k automatic ranking. Mutually exclusive with `by_path`; setting both raises `ValueError` at `__init__` and `set_params` time. Each path tuple must be a list/tuple of `int` of length `L_max + 1` (uniformity validated at `__init__`; length match against `L_max + 1` validated at fit-time); `bool` and `np.bool_` are explicitly rejected, `np.integer` accepted and canonicalized to Python `int` for tuple-key consistency. Duplicates emit a `UserWarning` and are deduplicated; paths not observed in the panel emit a `UserWarning` and are omitted from `path_effects`. Paths appear in `results.path_effects` in the user-specified order, modulo deduplication and unobserved-path filtering. Composes with non-binary D and all downstream `by_path` surfaces (bootstrap, per-path placebos, per-path joint sup-t bands, `controls`, `trends_linear`, `trends_nonparam`) — mechanical filter on observed paths via the same `_enumerate_treatment_paths` call site, no methodology change. **Python-only API extension; no R equivalent** — R's `did_multiplegt_dyn(..., by_path=k)` only accepts a positive int (top-k) or `-1` (all paths). The `by_path` precondition gate at `chaisemartin_dhaultfoeuille.py:1118` (drop_larger_lower / L_max / `heterogeneity` / `design2` / `honest_did` / `survey_design` mutex) and the 11 `self.by_path is not None` activation branches in `fit()` were rerouted to fire under either selector. Validation + behavior + cross-feature regressions at `tests/test_chaisemartin_dhaultfoeuille.py::TestPathsOfInterest`.
- **HAD `practitioner_next_steps()` handler + `llms-full.txt` reference section** (Phase 5). Adds `_handle_had` and `_handle_had_event_study` to `diff_diff/practitioner.py::_HANDLERS`, routing both `HeterogeneousAdoptionDiDResults` (single-period) and `HeterogeneousAdoptionDiDEventStudyResults` (event-study) through HAD-specific Baker et al. (2025) step guidance: `did_had_pretest_workflow` (step 3 — paper Section 4.2 step-2 closure on the event-study path), an estimand-difference routing nudge to `ContinuousDiD` (step 4 — fires when the user wants per-dose ATT(d) / ACRT(d) curves rather than HAD's WAS estimand and has never-treated controls; framed around estimand difference, NOT around the existence of untreated units, since HAD remains valid with a small never-treated share per REGISTRY § HeterogeneousAdoptionDiD edge cases and explicitly retains never-treated units on the staggered event-study path per paper Appendix B.2 / `had.py:1325`), `results.bandwidth_diagnostics` inspection on continuous designs and simultaneous (sup-t) `cband_*` reading on weighted event-study fits (step 6), per-horizon WAS event-study disaggregation (step 7), and the explicit design-auto-detection / last-cohort-only-WAS framing (step 8). Symmetric pair: `_handle_continuous` gains a Step-4 nudge to `HeterogeneousAdoptionDiD` for ContinuousDiD users on no-untreated panels (this direction is correct because ContinuousDiD's identification requires never-treated controls). Extends `_check_nan_att` with an ndarray branch via lazy `numpy` import for HAD's per-horizon `att` array; uses `np.all(np.isnan(arr))` semantics so partial-NaN arrays (legitimate event-study output under degenerate horizon-specific designs) do not over-fire the warning. Scalar path is bit-exact preserved across all 12 untouched handlers. Adds full HAD section + `HeterogeneousAdoptionDiDResults` / `HeterogeneousAdoptionDiDEventStudyResults` blocks + `## HAD Pretests` index covering all 7 pretest entry points + Choosing-an-Estimator row to `diff_diff/guides/llms-full.txt` (the bundled-in-wheel agent reference); the documented constructor + `fit()` signatures match the real `HeterogeneousAdoptionDiD.__init__` / `.fit` API exactly (verified by `inspect.signature`-based regression tests). Tightens the existing `Continuous treatment intensity` Choosing row to surface ATT(d) vs WAS as the estimand differentiator. `docs/doc-deps.yaml` updated to remove the `llms-full.txt` deferral note on `had.py` and add `llms-full.txt` entries to `had.py`, `had_pretests.py`, and `practitioner.py` blocks. Patch-level (additive on stable surfaces). 26 new tests (16 in `tests/test_practitioner.py::TestHADDispatch` + 9 in `tests/test_guides.py::TestLLMsFullHADCoverage` + 1 fixture-minimality regression locking the "handlers are STRING-ONLY at runtime" stability invariant). Closes the Phase 5 "agent surfaces" gap. T21 pretest tutorial subsequently landed in PR #409; T22 weighted/survey tutorial remains queued as a separate notebook PR.

### Changed
- **CI AI reviewer now sees tutorial notebook prose, with workflow hardening.** Replaced the blanket `:!docs/tutorials/*.ipynb` exclusion in the AI review pipeline with a markdown extraction step (`tools/notebook_md_extract.py`) that loops over changed tutorial notebooks and appends prose + code + executed outputs to the reviewer prompt under a new `## Tutorial Notebook Prose` section. The extractor uses stdlib `json` (no `nbformat` dep, no `pip install` step in the workflow) with a `_to_str()` helper that coerces nbformat raw JSON's list-or-string `source` / `text` fields. `text/html`-only outputs, `image/*` data, and `raw` cells are intentionally dropped (see `tools/notebook_md_extract.py` module docstring for the policy). **Workflow hardening:** the reviewer prompt (`.github/codex/prompts/pr_review.md`), the prompt-builder script (`.claude/scripts/openai_review.py`), AND the extractor (`tools/notebook_md_extract.py`) are all staged from `BASE_SHA` via `git show "$BASE_SHA:..." > /tmp/...` rather than the PR checkout, so a malicious PR cannot rewrite reviewer instructions, exfiltrate `OPENAI_API_KEY` via the script, or inject code into a step that runs before the secret-bearing API call. `actions/checkout@v6` uses `persist-credentials: false` and the pre-fetch step passes `GITHUB_TOKEN` via `http.extraheader` (env-scoped) to block secondary token exfiltration via `.git/config` reads. The PR-controlled notebook prose is wrapped in `<notebook-prose untrusted="true">` tags (mirroring PR #415's `<pr-body untrusted="true">` and `<previous-review-output untrusted="true">` conventions); the reviewer prompt instructs the model to ignore any directive inside the wrapper. The three wrapper-sanitization sites now share a single `_sanitize_wrapper_tag(text, tag_name)` helper in `openai_review.py` (refactored from the prior inline regex + `_sanitize_pr_body`; the latter is kept as a backward-compatible thin wrapper). Per-output cap `--max-output-chars 20000` and per-notebook cap `--max-total-chars 200000` keep the prompt within input budget on any single oversized output or any single oversized notebook. `tools/**` added to `rust-test.yml` path filters so future extractor-only changes still trigger the test job. Reaped the temporary T21 review aid at `docs/_review/t21_notebook_extract.md` and the `_review` entry in `docs/conf.py:exclude_patterns` (both lingered on `origin/main` from PR #409 — they should have been cleaned up there). Closes the gap surfaced during PR #409 (T21), where the reviewer ran 3+ rounds blind to the actual tutorial prose.

## [3.3.2] - 2026-04-26

### Added
- **`ChaisemartinDHaultfoeuille.by_path` is now compatible with `trends_linear` (DID^{fd} group-specific linear trends) and `trends_nonparam` (state-set trends).** For `trends_linear`, the first-differencing transform runs once globally before path enumeration, so per-path raw second-differences `DID^{fd}_{path, l}` surface on `path_effects[path]["horizons"][l]` automatically. Per-path **cumulated level effects** `delta_{path, l} = sum_{l'=1..l} DID^{fd}_{path, l'}` (the quantity R returns under `did_multiplegt_dyn(..., by_path, trends_lin)`) surface on the new `results.path_cumulated_event_study[path][l]` field, mirroring the global `linear_trends_effects` cumulation. `to_dataframe(level="by_path")` exposes `cumulated_effect` / `cumulated_se` columns (always present, NaN-when-None — mirrors the `cband_*` convention from PR #374); `summary()` renders a "Cumulated Level Effects (DID^{fd}, trends_linear)" sub-section under each per-path block. SE on the cumulated layer is the conservative upper bound (sum of per-horizon component SEs, NaN-consistent), matching the global `linear_trends_effects` convention. Path enumeration runs on the post-first-differenced `N_mat_fd`: switchers with `F_g==2` fail the window-eligibility check and are dropped from path enumeration entirely (the existing global `F_g >= 3` warning still surfaces the issue), so a path whose switchers all have `F_g < 3` is silently absent from `path_effects` rather than present-with-NaN. Placebo under `trends_linear` returns RAW per-horizon values — there is no per-path placebo cumulation surface in either Python or R. For `trends_nonparam`, the set membership column is validated and stored once globally as `set_ids_arr`; the `set_ids` parameter is now threaded through the four per-path IF helpers (`_compute_path_effects`, `_compute_path_placebos`, `_collect_path_bootstrap_inputs`, `_collect_path_placebo_bootstrap_inputs`) so per-path analytical SE, bootstrap, placebos, and sup-t bands all consume the set-restricted control pool automatically. Per-period effects remain unadjusted under both extensions, consistent with the existing per-period DID contract. Validated against R via two new golden-value scenarios: `single_baseline_multi_path_by_path_trends_lin` (n_periods=13, F_g >= 4, cohort-single-path; per-path cumulated point estimates match R bit-exactly with `POINT_RTOL=1e-9`, cumulated SE within `CUM_SE_RTOL=0.20`) and `multi_path_reversible_by_path_trends_nonparam` (per-path point estimates AND placebos match R bit-exactly with `POINT_RTOL=1e-9`, per-path SE within `SE_RTOL=0.15`). **F_g=3 boundary-case divergence (`by_path + trends_linear`):** `F_g=3` switchers have only 1 valid pre-window Z value after first-differencing, triggering 30%+ relative divergence between Python and R per-path point estimates on paths whose switchers include `F_g=3`. A targeted `UserWarning` fires at fit-time on this regime; R parity is asserted only on the `F_g >= 4` parity fixture. Placebo parity for `trends_linear` is intentionally skipped (R's per-path placebo computation re-runs on the path-restricted subsample with different control eligibility than Python's global-then-disaggregate architecture surfaces; placebo + `trends_linear` is exercised via internal regression only). Cross-path cohort-sharing SE deviation from R documented for `path_effects` is inherited unchanged. Gates at `chaisemartin_dhaultfoeuille.py:1014-1023` removed; `by_path` docstring updated to add the two new compatibility paragraphs and remove `trends_linear` / `trends_nonparam` from the incompatible list. R-parity tests at `tests/test_chaisemartin_dhaultfoeuille_parity.py::TestDCDHDynRParityByPathTrendsLinear` and `::TestDCDHDynRParityByPathTrendsNonparam`; cross-surface regressions at `tests/test_chaisemartin_dhaultfoeuille.py::TestByPathTrendsLinear` and `::TestByPathTrendsNonparam`. See `docs/methodology/REGISTRY.md` §ChaisemartinDHaultfoeuille `Note (Phase 3 by_path ...)` → "Per-path linear-trends DID^{fd}" and "Per-path state-set trends" for the full contract.
- **`yatchew_hr_test(null="mean_independence")` mode** mirroring R `YatchewTest::yatchew_test(order=0)`. Adds a `null: Literal["linearity", "mean_independence"]` keyword-only kwarg to `yatchew_hr_test`. Default `"linearity"` is bit-exact backcompat (residuals from OLS `dy = a + b·d + eps`, paper Assumption 8 / Theorem 7). New `"mean_independence"` fits intercept-only OLS (`dy = a + eps`, residuals `= dy - mean(dy)`); the downstream `sigma2_diff` / `sigma2_W` / sort-by-`d` machinery is identical between the two modes. Exposed on both unweighted and survey-weighted code paths (`weights=` / `survey_design=` compose orthogonally with `null=`). Adds a `null_form: str` field to `YatchewTestResults` so `summary()` renders the correct null-hypothesis description; `__repr__` and `to_dict()` updated. Closes the placebo Yatchew R-parity gap from PR #392 — `tests/test_did_had_parity.py::TestYatchewParity` now routes effect rows through `null="linearity"` (R `order=1`) and placebo rows through `null="mean_independence"` (R `order=0`); both modes share the documented `× G/(G-1)` finite-sample convention shift and parity holds at `atol=1e-10`. Patch-level (additive keyword-only kwarg + additive dataclass field with default).
- **HAD `trends_lin=True` linear-trend detrending mode** on `HeterogeneousAdoptionDiD.fit(aggregate="event_study")`, `joint_pretrends_test`, and `joint_homogeneity_test`. Mirrors R `DIDHAD::did_had(..., trends_lin=TRUE)` (paper Eq. 17 / Eq. 18 / page 32 joint-Stute homogeneity-with-trends). Per-group linear-trend slope estimated as `Y[g, F-1] - Y[g, F-2]` and applied as `(t - base) × slope` adjustment to per-event-time outcome evolutions. Requires F ≥ 3 (panel must contain F-2). The "consumed" placebo at our event-time `e=-2` is auto-dropped (R reduces max placebo lag by 1 with the same effect). Mutually exclusive with survey weighting (`survey_design` / `survey` / `weights`): raises `NotImplementedError` per `feedback_per_method_survey_element_contract` (weighted slope estimator not derived from paper; tracked in TODO.md as a follow-up). Bit-exact backcompat for `trends_lin=False` (default). Patch-level (additive keyword-only kwarg).
- **HAD R-package end-to-end parity test** vs `DIDHAD` v2.0.0 (`Credible-Answers/did_had`) on the **`design="continuous_at_zero"` (Design 1') surface**. New parity fixture `benchmarks/data/did_had_golden.json` generated by `benchmarks/R/generate_did_had_golden.R` covers 3 paper-derived synthetic DGPs (Uniform, Beta(2,2), Beta(0.5,1)) × 5 method combinations (overall, event-study, placebo, yatchew, trends_lin). The harness explicitly forces `HeterogeneousAdoptionDiD(design="continuous_at_zero")` because R `did_had` always evaluates the local-linear at `d=0` regardless of dose distribution; our default `design="auto"` may legitimately choose `continuous_near_d_lower` or `mass_point` on dose distributions with boundary density bounded away from zero (e.g., Beta(2,2)) and thereby diverge from R numerically — that divergence is methodologically defensible but out of scope for this parity test. Python parity test `tests/test_did_had_parity.py` asserts point estimate / SE / CI bounds at `atol=1e-8` and Yatchew T-stat at `atol=1e-10` after a documented `× G/(G-1)` finite-sample convention shift. Two intentional convention deviations from R, documented in `docs/methodology/REGISTRY.md`: (a) we report the bias-corrected point estimate (modern CCF 2018 convention; R's `Estimate` column reports the conventional estimate with the bias-corrected CI separately — our `att` matches R's CI midpoint); (b) Yatchew uses paper Appendix E's literal (1/G) variance-denominator convention while R uses base-R `var()`'s (1/(N-1)) sample-variance convention (parity is bit-exact after the `× G/(G-1)` shift). Yatchew on placebos with R's mean-independence null (`order=0`) was not exposed in `yatchew_hr_test` at the PR #392 cut and was skipped in the parity test; the follow-up `yatchew_hr_test(null="mean_independence")` entry above closes that gap (placebo rows now routed through `null="mean_independence"` and parity holds at the same `atol=1e-10`).
- **Tutorial 20: HAD for National Brand Campaign with Regional Spend Intensity** (`docs/tutorials/20_had_brand_campaign.ipynb`) — end-to-end practitioner walkthrough for `HeterogeneousAdoptionDiD` on a 60-DMA panel where every market is treated at a different dose level and no never-treated unit exists; comparison comes from dose variation across markets, not from an untreated holdout. The DGP uses Uniform[\$5K, \$50K] regional add-on spend per DMA (every DMA participates, no DMA at exactly \$0), so `design="auto"` resolves to `continuous_near_d_lower` (Design 1) with target `WAS_d_lower` — interpreted as the average per-dollar marginal effect of regional spend above the lightest-touch DMA's spend (`d_lower` ≈ \$5K). Covers the headline `WAS_d_lower` fit on a 2-period collapse, the multi-week event study with per-week pointwise CIs and pre-launch placebos, and a stakeholder communication template that flags the Assumption 5/6 caveat (non-testable local-linearity at the boundary). Companion drift-test file `tests/test_t20_had_brand_campaign_drift.py` (13 tests pinning panel composition / sample median, design auto-detection / target / `d_lower`, overall `WAS_d_lower`, CI endpoints, dose mean, n_units, full event-study horizon presence, and per-horizon coverage). T20 wired into the existing `had.py` entry in `docs/doc-deps.yaml`; cross-link added from `docs/practitioner_decision_tree.rst` § "Universal Rollout (No Untreated Markets)" via a `.. tip::` block.

### Changed
- **Rust dependency upgrades**: bumped `rand` 0.8 → 0.10 and `rand_xoshiro` 0.6 → 0.8 in the Rust backend (the two crates are coupled through `rand_core` and must move together). MSRV bumped from Rust 1.84 → 1.85 to satisfy the new dependency requirements. Three call sites in `rust/src/bootstrap.rs` updated for the `rand 0.9` API rename: `gen::<bool>()` → `random::<bool>()`, `gen::<f64>()` → `random::<f64>()`, `gen_range(0..6)` → `random_range(0..6)`. **Webb wild bootstrap byte stream shifted** as a side effect: `rand 0.9` reworked the internal algorithm for `random_range` (improved rejection sampling), so `Xoshiro256PlusPlus::seed_from_u64(seed)` followed by `random_range(0..6)` consumes RNG bytes differently than the old `gen_range(0..6)` did. Distributional properties of Webb weights are unchanged (still uniform over the 6-point support); aggregate inference (SE, p-values, CI) converges to the same values for any reasonable `n_bootstrap`. Rademacher and Mammen byte streams are bit-identical to the prior release. Anyone with a saved Rust+Webb baseline pinning specific seeded results will see different numbers; the regression test suite uses within-build seed-reproducibility (not cross-version baselines) so all internal tests pass unchanged. New regression guard `TestRustBackend::test_bootstrap_weights_bit_identity_snapshot` pins fixed-seed weights for all three weight types, so any future RNG drift fails loudly with a localized error message.

## [3.3.1] - 2026-04-25

### Changed
- **HAD survey-design API consolidated to single `survey_design=` kwarg** across all 8 HAD surfaces: `HeterogeneousAdoptionDiD.fit`, `did_had_pretest_workflow`, `qug_test`, `stute_test`, `yatchew_hr_test`, `stute_joint_pretest`, `joint_pretrends_test`, `joint_homogeneity_test`. Matches the rest of the library (`ContinuousDiD`, `EfficientDiD`, `ChaisemartinDHaultfoeuille` already used `survey_design=`). On data-in surfaces (HAD.fit, workflow, joint data-in wrappers) `survey_design=` accepts a `SurveyDesign` instance (column references resolved against `data` at fit time, same convention as the rest of the library). On the three array-in linearity helpers (`stute_test`, `yatchew_hr_test`, `stute_joint_pretest`) `survey_design=` accepts a pre-resolved `ResolvedSurveyDesign`; passing a `SurveyDesign` raises `TypeError` with migration guidance to `make_pweight_design(arr)` (pweight-only) or pre-resolution. `qug_test` is the 8th surface and accepts the same kwarg signature for consistency, but **all** non-`None` values raise `NotImplementedError` per the Phase 4.5 C0 permanent deferral (no migration path; the qug-specific mutex error reflects this). New public helper `make_pweight_design(weights: np.ndarray) -> ResolvedSurveyDesign` exported from the `diff_diff` top level for the pweight-only convenience on the three array-in linearity helpers (formerly the private `survey._make_trivial_resolved`, kept as a permanent private alias); validates 1-D input at the front door. Three-way mutex (`survey_design + survey + weights`) extends the prior 2-way (`survey + weights`) — at most one may be non-None per call. Patch-level addition (additive new kwarg + permanent alias for the helper; no breaking changes this release).

### Deprecated
- **`HeterogeneousAdoptionDiD.fit(survey=, weights=)`, `did_had_pretest_workflow(survey=, weights=)`, and the 6 HAD pretest helpers' `survey=` / `weights=` kwargs are deprecated** in favor of the canonical `survey_design=`. Emits `DeprecationWarning` with migration guidance; the deprecated kwargs continue to route through the unchanged legacy back-end paths so numerical results are identical to pre-PR (bit-exact regression locked by parity tests in `tests/test_had_dual_knob_deprecation.py`). Both `survey=` and `weights=` will be removed in the next minor release. **Carve-out for `qug_test`**: the deprecation is kwarg-name-consolidation only; `qug_test` permanently rejects all non-`None` `survey_design` / `survey` / `weights` values (Phase 4.5 C0 deferral) and `make_pweight_design(arr)` is NOT a valid migration target — the deprecation warning text on `qug_test` is qug-specific and points users to `did_had_pretest_workflow(..., survey_design=...)` for survey-aware HAD pretesting (which skips the QUG step under survey).

### Added
- **`ChaisemartinDHaultfoeuille.by_path` + `controls`** (DID^X residualization) — the per-baseline OLS residualization (Web Appendix Section 1.2) is now compatible with `by_path=k`. The residualization runs once on the first-differenced outcome BEFORE path enumeration, so all four downstream surfaces (analytical per-path SE, bootstrap SE, per-path placebos, per-path joint sup-t bands) consume the residualized `Y_mat` automatically (Frisch-Waugh-Lovell). Per-period effects remain unadjusted, consistent with the existing `controls` + per-period DID contract (per-period DID does not support residualization). Failed-stratum baselines (rank-deficient X) zero out `N_mat` for affected groups, which the path enumeration treats as ineligible per its existing convention. **Deviation from R on multi-baseline switcher panels (point estimates):** R `did_multiplegt_dyn(..., by_path, controls)` re-runs the per-baseline OLS residualization on each path's restricted subsample (path's switchers + same-baseline not-yet-treated controls), so its residualization coefficients vary per path when switchers have different baseline values. Our global-residualization architecture coincides with R on single-baseline switcher panels (every switcher shares the same `D_{g,1}`) — per-path point estimates match R exactly there. On multi-baseline panels, point estimates can diverge; the estimator emits a `UserWarning` at fit-time when this configuration is detected so practitioners do not silently consume estimates that disagree with R. **SE inherits the cross-path cohort-sharing SE deviation from R** documented for `path_effects` — bootstrap SE, placebo SE, and sup-t crit are Monte Carlo / joint-distribution analogs of the same residualized analytical IF and carry the same deviation. R-parity confirmed against `did_multiplegt_dyn(..., by_path=3, controls="X1")` via the new `multi_path_reversible_by_path_controls` single-baseline golden-value scenario (per-path point estimates match R bit-exactly — measured rtol ~1e-11 across all path × horizon cells — on this one-observation-per-cell scenario; per-path SE within ~6.5% of R, well inside the Phase 2 multi-horizon envelope). Cell-aggregated panels with multiple observations per `(g, t)` also coincide with our equal-cell-weighting first stage rather than R's `N_gt`-weighted first stage per the existing DID^X cell-weighting deviation documented in `docs/methodology/REGISTRY.md` `Note (Phase 3 DID^X covariate adjustment)`. Gate at `chaisemartin_dhaultfoeuille.py:988-992` removed; `by_path` docstring updated to add the new compatibility paragraph (with the multi-baseline caveat) and remove `controls` from the incompatible list. R-parity test at `tests/test_chaisemartin_dhaultfoeuille_parity.py::TestDCDHDynRParityByPathControls`; cross-surface inheritance + multi-baseline `UserWarning` regression-tested at `tests/test_chaisemartin_dhaultfoeuille.py::TestByPathControls` (analytical + bootstrap + placebo + sup-t + `to_dataframe(level="by_path")` cband columns + multi-baseline warning). See `docs/methodology/REGISTRY.md` §ChaisemartinDHaultfoeuille `Note (Phase 3 by_path ...)` → "Per-path covariate residualization (DID^X)" for the full contract.
- **HAD linearity-family pretests under survey (Phase 4.5 C).** `stute_test`, `yatchew_hr_test`, `stute_joint_pretest`, `joint_pretrends_test`, `joint_homogeneity_test`, and `did_had_pretest_workflow` now accept `weights=` / `survey=` keyword-only kwargs. Stute family uses **PSU-level Mammen multiplier bootstrap** via `bootstrap_utils.generate_survey_multiplier_weights_batch` (the same kernel as PR #363's HAD event-study sup-t bootstrap): each replicate draws an `(n_bootstrap, n_psu)` Mammen multiplier matrix, broadcast to per-obs perturbation `eta_obs[g] = eta_psu[psu(g)]`, weighted OLS refit, weighted CvM via new `_cvm_statistic_weighted` helper. Joint Stute SHARES the multiplier matrix across horizons within each replicate, preserving both the vector-valued empirical-process unit-level dependence AND PSU clustering. Yatchew uses **closed-form weighted OLS + pweight-sandwich variance components** (no bootstrap): `sigma2_lin = sum(w·eps²)/sum(w)`, `sigma2_diff = sum(w_avg·diff²)/(2·sum(w))` with arithmetic-mean pair weights `w_avg_g = (w_g+w_{g-1})/2`, `sigma4_W = sum(w_avg·prod)/sum(w_avg)`, `T_hr = sqrt(sum(w))·(sigma2_lin-sigma2_diff)/sigma2_W`. All three Yatchew components reduce bit-exactly to the unweighted formulas at `w=ones(G)` (locked at `atol=1e-14` by direct helper test). The pweight `weights=` shortcut routes through a synthetic trivial `ResolvedSurveyDesign` (new `survey._make_trivial_resolved` helper) so the same kernel handles both entry paths. `did_had_pretest_workflow(..., survey=, weights=)` removes the Phase 4.5 C0 `NotImplementedError`, dispatches to the survey-aware sub-tests, **skips the QUG step with `UserWarning`** (per C0 deferral), sets `qug=None` on the report, and appends a `"linearity-conditional verdict; QUG-under-survey deferred per Phase 4.5 C0"` suffix to the verdict. `HADPretestReport.qug` retyped from `QUGTestResults` to `Optional[QUGTestResults]`; `summary()` / `to_dict()` / `to_dataframe()` updated to None-tolerant rendering. Replicate-weight survey designs (BRR/Fay/JK1/JKn/SDR) raise `NotImplementedError` at every entry point (defense in depth, reciprocal-guard discipline) — parallel follow-up after this PR. **Stratified designs (`SurveyDesign(strata=...)`) also raise `NotImplementedError` on the Stute family** — the within-stratum demean + `sqrt(n_h/(n_h-1))` correction that the HAD sup-t bootstrap applies to match the Binder-TSL stratified target has not been derived for the Stute CvM functional, so applying raw multipliers from `generate_survey_multiplier_weights_batch` directly to residual perturbations would leave the bootstrap p-value silently miscalibrated. Phase 4.5 C narrows survey support to **pweight-only**, **PSU-only** (`SurveyDesign(weights=, psu=)`), and **FPC-only** (`SurveyDesign(weights=, fpc=)`) designs; stratified is a follow-up after the matching Stute-CvM stratified-correction derivation lands. Strictly positive weights required on Yatchew (the adjacent-difference variance is undefined under contiguous-zero blocks). Per-row `weights=` / `survey=col` aggregated to per-unit via existing HAD helpers `_aggregate_unit_weights` / `_aggregate_unit_resolved_survey` (constant-within-unit invariant enforced). Unweighted code paths preserved bit-exactly. Patch-level addition (additive on stable surfaces). See `docs/methodology/REGISTRY.md` § "QUG Null Test" — Note (Phase 4.5 C) for the full methodology.
- **`ChaisemartinDHaultfoeuille.by_path` + `n_bootstrap > 0` joint sup-t bands** — per-path joint sup-t simultaneous confidence intervals across horizons `1..L_max` within each path. A single shared `(n_bootstrap, n_eligible)` multiplier weight matrix (using the estimator's configured `bootstrap_weights` — Rademacher / Mammen / Webb) is drawn per path and broadcast across all horizons of that path, producing correlated bootstrap distributions across horizons. The path-specific critical value `c_p = quantile(max_l |t_l|, 1 - α)` is used to construct symmetric joint bands `effect_l ± c_p · se_l` per horizon. Surfaced on `results.path_sup_t_bands` (dict keyed by path tuple, each entry with `crit_value / alpha / n_bootstrap / method / n_valid_horizons`); as `cband_conf_int` per horizon entry on `path_effects[path]["horizons"][l]`; and as `cband_lower` / `cband_upper` columns on `results.to_dataframe(level="by_path")` (mirrors the OVERALL `level="event_study"` schema; positive-horizon rows of banded paths get populated values, placebo / unbanded / empty-window rows get NaN). Gates: a path needs `>= 2` valid horizons (finite bootstrap SE > 0) AND a strict majority (more than 50%) of finite sup-t draws to receive a band. Empty-state contract: `path_sup_t_bands is None` when not requested; `{}` when requested but no path passes both gates. **Methodology asymmetry vs OVERALL `event_study_sup_t_bands`:** the per-path sup-t draws a fresh shared weight matrix per path AFTER the per-path SE bootstrap block has already populated `results.path_ses` via independent per-(path, horizon) draws — asymptotically equivalent to OVERALL's self-consistent reuse but NOT bit-identical. Documented intentional choice to preserve RNG-state isolation for existing per-path SE seed-reproducibility tests. Inherits the cross-path cohort-sharing SE deviation from R documented for `path_effects`. **Deviation from R:** `did_multiplegt_dyn` does not provide joint / sup-t bands at any surface — this is a Python-only methodology extension consistent with the existing OVERALL sup-t bands (also Python-only). Bands cover joint inference WITHIN a single path across horizons; they do NOT provide simultaneous coverage across paths. Pre-audit fix bundled: stale "Phase 2 placeholder" docstring on the existing `sup_t_bands` field updated to the actual contract description. Tests at `tests/test_chaisemartin_dhaultfoeuille.py::TestByPathSupTBands` (`@pytest.mark.slow`). See `docs/methodology/REGISTRY.md` §ChaisemartinDHaultfoeuille `Note (Phase 3 by_path per-path joint sup-t bands)` for the full contract.
- **`ChaisemartinDHaultfoeuille.by_path` + `placebo=True`** — per-path backward-horizon placebos `DID^{pl}_{path, l}` for `l = 1..L_max`. The same per-path SE convention used for the event-study (joiners/leavers IF precedent: switcher-side contributions zeroed for non-path groups; cohort structure and control pool unchanged; plug-in SE with path-specific divisor `N^{pl}_{l, path}`) is applied to backward horizons via the new `switcher_subset_mask` parameter on `_compute_per_group_if_placebo_horizon`. Surfaced on `results.path_placebo_event_study[path][-l]` (negative-int inner keys mirroring `placebo_event_study`); `summary()` renders the rows alongside per-path event-study horizons; `to_dataframe(level="by_path")` emits negative-horizon rows alongside the existing positive-horizon rows. **Bootstrap** (when `n_bootstrap > 0`) propagates per-`(path, lag)` percentile CI / p-value through the same `_bootstrap_one_target` dispatch as the per-path event-study, with the canonical NaN-on-invalid contract enforced on the new surface (PR #364 library-wide invariant). **SE inherits the cross-path cohort-sharing deviation from R** documented for `path_effects` (full-panel cohort-centered plug-in vs R's per-path re-run): tracks R within tolerance on single-path-cohort panels, diverges materially on cohort-mixed panels — the bootstrap SE is a Monte Carlo analog of the analytical SE and inherits the same deviation. R-parity confirmed at `tests/test_chaisemartin_dhaultfoeuille_parity.py::TestDCDHDynRParityByPathPlacebo` on the new `multi_path_reversible_by_path_placebo` scenario (point estimates exact match; SE within Phase-2 envelope rtol ≤ 5%); positive analytical + bootstrap invariants at `tests/test_chaisemartin_dhaultfoeuille.py::TestByPathPlacebo` (and the gated `::TestBootstrap` subclass). See `docs/methodology/REGISTRY.md` §ChaisemartinDHaultfoeuille `Note (Phase 3 by_path ...)` → "Per-path placebos" for the full contract.
- **Tutorial 19: dCDH for Marketing Pulse Campaigns** (`docs/tutorials/19_dcdh_marketing_pulse.ipynb`) — end-to-end practitioner walkthrough on a 60-market reversible-treatment panel covering the TWFE decomposition diagnostic (`twowayfeweights`), `DCDH` Phase 1 (DID_M, joiners-vs-leavers, single-lag placebo), the `L_max` multi-horizon event study with multiplier bootstrap, a stakeholder communication template, and drift guards. README listing for Tutorial 17 (Brand Awareness Survey) backfilled in the same edit. Cross-link from `docs/practitioner_decision_tree.rst` § "Reversible Treatment" added.

## [3.3.0] - 2026-04-25

### Fixed
- **`SyntheticDiD(variance_method="placebo")` SE now uses R-default warm-start** matching `synthdid:::placebo_se`. R's placebo loop seeds Frank-Wolfe per draw with `weights.boot$omega = sum_normalize(weights$omega[ind[1:N0_placebo]])` (fit-time ω subsetted + renormalized) and the fit-time `weights$lambda` — Python previously used uniform cold-start, producing finite-iter convergence-pattern drift on a handful of draws relative to R's reference SE. New `_placebo_variance_se` kwargs `init_omega` / `init_lambda` thread fit-time weights through the existing two-pass FW dispatcher; on the global FW optimum the values are init-independent (strictly convex objective), so the change is a finite-iter parity fix, not a methodology change. Existing placebo SE values shift by sub-percent on most panels; the bit-identity baseline pin in `TestScaleEquivariance::test_baseline_parity_small_scale[placebo]` was rebased from `0.29385822261006445` to `0.293840360160448`. New R-parity test `tests/test_methodology_sdid.py::TestJackknifeSERParity::test_placebo_se_matches_r` asserts SE matches R's `vcov(method="placebo")` to within `< 1e-8` using R's exact permutation sequence (recorded by `benchmarks/R/generate_sdid_placebo_parity_fixture.R` into `tests/data/sdid_placebo_indices_r.json`). The `_placebo_indices` kwarg on `_placebo_variance_se` is the test seam; not part of the public API.

### Added
- **`qug_test` and `did_had_pretest_workflow` survey-aware NotImplementedError gates (Phase 4.5 C0 decision gate).** `qug_test(d, *, survey=None, weights=None)` and `did_had_pretest_workflow(..., *, survey=None, weights=None)` now accept the two kwargs as keyword-only with default `None`. Passing either non-`None` raises `NotImplementedError` with an educational message naming the methodology rationale and pointing users to joint Stute (Phase 4.5 C, planned) as the survey-compatible alternative. Mutex guard on `survey=` + `weights=` mirrors `HeterogeneousAdoptionDiD.fit()` at `had.py:2890`. **QUG-under-survey is permanently deferred** — the test statistic uses extreme order statistics `D_{(1)}, D_{(2)}` which are NOT smooth functionals of the empirical CDF, so standard survey machinery (Binder-TSL linearization, Rao-Wu rescaled bootstrap, Krieger-Pfeffermann (1997) EDF tests) does not yield a calibrated test; under cluster sampling the `Exp(1)/Exp(1)` limit law's independence assumption breaks; and the EVT-under-unequal-probability-sampling literature (Quintos et al. 2001, Beirlant et al.) addresses tail-index estimation, not boundary tests. The workflow's gate is **temporary** — Phase 4.5 C will close it for the linearity-family pretests with mechanism varying by test: Rao-Wu rescaled bootstrap for `stute_test` and the joint variants (`stute_joint_pretest`, `joint_pretrends_test`, `joint_homogeneity_test`); weighted OLS residuals + weighted variance estimator for `yatchew_hr_test` (Yatchew 1997 is a closed-form variance-ratio test, not bootstrap-based). Sister pretests (`stute_test`, `yatchew_hr_test`, `stute_joint_pretest`, `joint_pretrends_test`, `joint_homogeneity_test`) keep their closed signatures in this release — Phase 4.5 C will add kwargs and implementation together to avoid API churn. Unweighted `qug_test(d)` and `did_had_pretest_workflow(...)` calls are bit-exact pre-PR (kwargs are keyword-only after `*`; positional path unchanged). New tests at `tests/test_had_pretests.py::TestQUGTest` (5 rejection / mutex / message / regression tests) and the new `TestHADPretestWorkflowSurveyGuards` class (6 tests covering both kwarg paths, mutex, methodology pointer, both aggregate paths, and unweighted regression). See `docs/methodology/REGISTRY.md` § "QUG Null Test" — Note (Phase 4.5 C0) for the full methodology rationale plus a sketch of the (out-of-scope) theoretical bridge that combines endpoint-estimation EVT (Hall 1982, Aarssen-de Haan 1994, Hall-Wang 1999, Beirlant-de Wet-Goegebeur 2006), survey-aware functional CLTs (Boistard-Lopuhaä-Ruiz-Gazen 2017, Bertail-Chautru-Clémençon 2017), and tail-empirical-process theory (Drees 2003) — publishable methodology research, not engineering work.
- **`HeterogeneousAdoptionDiD` mass-point `survey=` / `weights=` + event-study `aggregate="event_study"` survey composition + multiplier-bootstrap sup-t simultaneous confidence band (Phase 4.5 B).** Closes the two Phase 4.5 A `NotImplementedError` gates: `design="mass_point" + weights/survey` and `aggregate="event_study" + weights/survey`. Weighted 2SLS sandwich in `_fit_mass_point_2sls` follows the Wooldridge 2010 Ch. 12 pweight convention (`w²` in the HC1 meat, `w·u` in the CR1 cluster score, weighted bread `Z'WX`); HC1 and CR1 ("stata" `se_type`) bit-parity with `estimatr::iv_robust(..., weights=, clusters=)` at `atol=1e-10` (new cross-language golden at `benchmarks/data/estimatr_iv_robust_golden.json`, generated by `benchmarks/R/generate_estimatr_iv_robust_golden.R`; `estimatr` added to `benchmarks/R/requirements.R`). `_fit_mass_point_2sls` gains `weights=` + `return_influence=` kwargs and now always returns a 3-tuple `(beta, se, psi)` — `psi` is the per-unit IF on the β̂-scale scaled so `compute_survey_if_variance(psi, trivial_resolved) ≈ V_HC1[1,1]` at `atol=1e-10` (PR #359 IF scale convention applied uniformly; no `sum(psi²)` claims). Event-study per-horizon variance: `survey=` path composes Binder-TSL via `compute_survey_if_variance`; `weights=` shortcut uses the analytical weighted-robust SE (continuous: CCT-2014 `bc_fit.se_robust / |den|`; mass-point: weighted 2SLS pweight sandwich from `_fit_mass_point_2sls` — HC1 / classical / CR1). `survey_metadata` / `variance_formula` / `effective_dose_mean` populated in both regimes (previously hardcoded `None` at `had.py:3366`). New multiplier-bootstrap sup-t: `_sup_t_multiplier_bootstrap` reuses `diff_diff.bootstrap_utils.generate_survey_multiplier_weights_batch` for PSU-level draws with stratum centering + sqrt(n_h/(n_h-1)) small-sample correction + FPC scaling + lonely-PSU handling. On the `weights=` shortcut, sup-t calibration is routed through a synthetic trivial `ResolvedSurveyDesign` so the centered + small-sample-corrected branch fires uniformly — targets the analytical HC1 variance family (`compute_survey_if_variance(IF, trivial) ≈ V_HC1` per the PR #359 IF scale invariant) rather than the raw `sum(ψ²) = ((n-1)/n) · V_HC1` that unit-level Rademacher multipliers would produce on the HC1-scaled IF. Perturbations: `delta = weights @ IF` with NO `(1/n)` prefactor (matching `staggered_bootstrap.py:373` idiom), normalized by per-horizon analytical SE, `(1-alpha)`-quantile of the sup-t distribution. At H=1 the quantile reduces to `Φ⁻¹(1 − alpha/2) ≈ 1.96` up to MC noise (regression-locked by `TestSupTReducesToNormalAtH1`). `HeterogeneousAdoptionDiD.__init__` gains `n_bootstrap: int = 999` and `seed: Optional[int] = None` (CS-parity singular seed); `fit()` gains `cband: bool = True` (only consulted on weighted event-study). `HeterogeneousAdoptionDiDEventStudyResults` extended with `variance_formula`, `effective_dose_mean`, `cband_low`, `cband_high`, `cband_crit_value`, `cband_method`, `cband_n_bootstrap` (all `None` on unweighted fits); surfaced in `to_dict`, `to_dataframe`, `summary`, `__repr__`. Unweighted event-study with `cband=False` preserves pre-Phase 4.5 B numerical output bit-exactly (stability invariant, locked by regression tests). Zero-weight subpopulation convention carries over from PR #359 (filter for design decisions; preserve full ResolvedSurveyDesign for variance). Non-pweight SurveyDesigns (`aweight`, `fweight`, replicate designs) raise `NotImplementedError` on both new paths (reciprocal-guard discipline). Pretest surfaces (`qug_test`, `stute_test`, `yatchew_hr_test`, joint variants, `did_had_pretest_workflow`) remain unweighted in this release — Phase 4.5 C / C0. See `docs/methodology/REGISTRY.md` §HeterogeneousAdoptionDiD "Weighted 2SLS (Phase 4.5 B)", "Event-study survey composition", and "Sup-t multiplier bootstrap" for derivations and invariants.
- **`PanelProfile.outcome_shape` and `PanelProfile.treatment_dose` extensions + `llms-autonomous.txt` worked examples (Wave 2 of the AI-agent enablement track).** `profile_panel(...)` now populates two new optional sub-dataclasses on the returned `PanelProfile`: `outcome_shape: Optional[OutcomeShape]` (numeric outcomes only — exposes `n_distinct_values`, `pct_zeros`, `value_min` / `value_max`, `skewness` and `excess_kurtosis` (NaN-safe; `None` when `n_distinct_values < 3` or variance is zero), `is_integer_valued`, `is_count_like` (heuristic: integer-valued AND has zeros AND right-skewed AND > 2 distinct values AND non-negative support, i.e. `value_min >= 0`; flags WooldridgeDiD QMLE consideration over linear OLS — the non-negativity clause aligns the routing signal with `WooldridgeDiD(method="poisson")`'s hard rejection of negative outcomes at `wooldridge.py:1105-1109`), `is_bounded_unit` ([0, 1] support)) and `treatment_dose: Optional[TreatmentDoseShape]` (continuous treatments only — exposes `n_distinct_doses`, `has_zero_dose`, `dose_min` / `dose_max` / `dose_mean` over non-zero doses). Both `OutcomeShape` and `TreatmentDoseShape` are mostly descriptive context. **`profile_panel` does not see the separate `first_treat` column** that `ContinuousDiD.fit()` consumes; the estimator's actual fit-time gates key off `first_treat` (defines never-treated controls as `first_treat == 0`, force-zeroes nonzero `dose` on those rows with a `UserWarning`, and rejects negative dose only among treated units `first_treat > 0`; see `continuous_did.py:276-327` and `:348-360`). In the canonical `ContinuousDiD` setup (Callaway, Goodman-Bacon, Sant'Anna 2024), the dose `D_i` is **time-invariant per unit** and `first_treat` is a **separate column** the caller supplies (not derived from the dose column). Under that setup, several facts on the dose column predict `fit()` outcomes: `PanelProfile.has_never_treated` (proxies `P(D=0) > 0` because the canonical convention ties `first_treat == 0` to `D_i == 0`); `PanelProfile.treatment_varies_within_unit == False` (the actual fit-time gate at line 222-228, holds regardless of `first_treat`); `PanelProfile.is_balanced` (the actual fit-time gate at line 329-338); absence of the `duplicate_unit_time_rows` alert (silent last-row-wins overwrite, must deduplicate before fit); and `treatment_dose.dose_min > 0` (predicts the strictly-positive-treated-dose requirement at line 287-294 because treated units carry their constant dose across all periods). When `has_never_treated == False` (no zero-dose controls but all observed doses non-negative), `ContinuousDiD` does not apply (Remark 3.1 lowest-dose-as-control is not implemented); `HeterogeneousAdoptionDiD` IS a routing alternative on this branch (HAD's own contract requires non-negative dose, which is satisfied). When `dose_min <= 0` (negative treated doses), `ContinuousDiD` does not apply AND `HeterogeneousAdoptionDiD` is **not** a fallback — HAD also raises on negative post-period dose (`had.py:1450-1459`); the applicable alternative is linear DiD with the treatment as a signed continuous covariate. Re-encoding the treatment column is an agent-side preprocessing choice that changes the estimand and is not documented in REGISTRY as a supported fallback. The estimator's force-zero coercion on `first_treat == 0` rows with nonzero `dose` is implementation behavior for inconsistent inputs, not a documented method for manufacturing never-treated controls. The agent must validate the supplied `first_treat` column independently — `profile_panel` does not see it. The shape extensions provide distributional context (effect-size range, count-shape detection) that supplements but does not replace those gates. Both fields are `None` when their classification gate is not met (e.g., `treatment_dose is None` for binary treatments). `to_dict()` serializes the nested dataclasses as JSON-compatible nested dicts. New exports: `OutcomeShape`, `TreatmentDoseShape` from top-level `diff_diff`. `llms-autonomous.txt` gains a new §5 "Worked examples" section with three end-to-end PanelProfile -> reasoning -> validation walkthroughs (binary staggered with never-treated controls, continuous dose with zero baseline, count-shaped outcome) plus §2 field-reference subsections for the new shape fields and §4.7 / §4.11 cross-references for outcome-shape considerations. Existing §5-§8 of the autonomous guide are renumbered to §6-§9. Descriptive only — no recommender language inside the worked examples.
- **`HeterogeneousAdoptionDiD.fit(survey=..., weights=...)` on continuous-dose paths (Phase 4.5 survey support).** The `continuous_at_zero` (paper Design 1') and `continuous_near_d_lower` (Design 1 continuous-near-d̲) designs accept survey weights through two interchangeable kwargs: `weights=<array>` (pweight shortcut, weighted-robust SE from the CCT-2014 lprobust port) and `survey=SurveyDesign(weights, strata, psu, fpc)` (design-based inference via Binder-TSL variance using the existing `compute_survey_if_variance` helper at `diff_diff/survey.py:1802`). Point estimates match across both entry paths; SE diverges by design (pweight-only vs PSU-aggregated). `HeterogeneousAdoptionDiDResults.survey_metadata` is a repo-standard `SurveyMetadata` dataclass (weight_type / effective_n / design_effect / sum_weights / weight_range / n_strata / n_psu / df_survey); HAD-specific extras (`variance_formula` label, `effective_dose_mean`) are separate top-level result fields. `to_dict()` surfaces the full `SurveyMetadata` object plus `variance_formula` + `effective_dose_mean`; `summary()` renders `variance_formula`, `effective_n`, `effective_dose_mean`, and (when the survey= path is used) `df_survey`; `__repr__` surfaces `variance_formula` + `effective_dose_mean` when present. The HAD `mass_point` design and `aggregate="event_study"` path raise `NotImplementedError` under survey/weights (deferred to Phase 4.5 B: weighted 2SLS + event-study survey composition); the HAD pretests stay unweighted in this release (Phase 4.5 C). Parity ceiling acknowledged — no public weighted-CCF bias-corrected local-linear reference exists in any language; methodology confidence comes from (1) uniform-weights bit-parity at `atol=1e-14` on the full lprobust output struct, (2) cross-language weighted-OLS parity (manual R reference) at `atol=1e-12`, and (3) Monte Carlo oracle consistency on known-τ DGPs. `_nprobust_port.lprobust` gains `weights=` and `return_influence=` (used internally by the Binder-TSL path); `bias_corrected_local_linear` removes the Phase 1c `NotImplementedError` on `weights=` and forwards. Auto-bandwidth selection remains unweighted in this release — pass `h`/`b` explicitly for weight-aware bandwidths. See `docs/methodology/REGISTRY.md` §HeterogeneousAdoptionDiD "Weighted extension (Phase 4.5 survey support)".
- **`stute_joint_pretest`, `joint_pretrends_test`, `joint_homogeneity_test` + `StuteJointResult`** (HeterogeneousAdoptionDiD Phase 3 follow-up). Joint Cramér-von Mises pretests across K horizons with shared-η Mammen wild bootstrap (preserves vector-valued empirical-process unit-level dependence per Delgado-Manteiga 2001 / Hlávka-Hušková 2020). The core `stute_joint_pretest` is residuals-in; two thin data-in wrappers construct per-horizon residuals for the two nulls the paper spells out: mean-independence (step 2 pre-trends, `OLS(Y_t − Y_base ~ 1)` per pre-period) and linearity (step 3 joint, `OLS(Y_t − Y_base ~ 1 + D)` per post-period). Sum-of-CvMs aggregation (`S_joint = Σ_k S_k`); per-horizon scale-invariant exact-linear short-circuit. Closes the paper Section 4.2 step-2 gap that Phase 3 `did_had_pretest_workflow` previously flagged with an "Assumption 7 pre-trends test NOT run" caveat. See `docs/methodology/REGISTRY.md` §HeterogeneousAdoptionDiD "Joint Stute tests" for algorithm, invariants, and scope exclusion of Eq 18 linear-trend detrending (deferred to Phase 4 Pierce-Schott replication).
- **`did_had_pretest_workflow(aggregate="event_study")`**: multi-period dispatch on balanced ≥3-period panels. Runs QUG at `F` + joint pre-trends Stute across earlier pre-periods + joint homogeneity-linearity Stute across post-periods. Step 2 closure requires ≥2 pre-periods; with only a single pre-period (the base `F-1`) `pretrends_joint=None` and the verdict flags the skip. Reuses the Phase 2b event-study panel validator (last-cohort auto-filter under staggered timing with `UserWarning`; `ValueError` when `first_treat_col=None` and the panel is staggered). The data-in wrappers `joint_pretrends_test` and `joint_homogeneity_test` also route through that same validator internally, so direct wrapper calls inherit the last-cohort filter and constant-post-dose invariant. `HADPretestReport` extended with `pretrends_joint`, `homogeneity_joint`, and `aggregate` fields; serialization methods (`summary`, `to_dict`, `to_dataframe`, `__repr__`) preserve the Phase 3 output bit-exactly on `aggregate="overall"` — no `aggregate` key, no header row, no schema drift — and only surface the new fields on `aggregate="event_study"`.
- **`ChaisemartinDHaultfoeuille.by_path`** — per-path event-study disaggregation, mirroring R `did_multiplegt_dyn(..., by_path=k)`. Passing `by_path=k` (positive int) to the estimator reports separate `DID_{path,l}` + SE + inference for the top-k most common observed treatment paths in the window `[F_g-1, F_g-1+L_max]`, answering the practitioner question "is a single pulse enough, or do you need sustained exposure?" across paths like `(0,1,0,0)` vs `(0,1,1,0)` vs `(0,1,1,1)`. The per-path SE follows the joiners-only / leavers-only IF precedent (switcher-side contribution zeroed for non-path groups; control pool and cohort structure unchanged; plug-in SE with path-specific divisor). Requires `drop_larger_lower=False` (multi-switch groups are the object of interest) and `L_max >= 1`. Binary treatment only in this release; combinations with `trends_linear`, `trends_nonparam`, `heterogeneity`, `design2`, `honest_did`, and `survey_design` raise `NotImplementedError` and are deferred to follow-up PRs (`n_bootstrap > 0`, `placebo=True`, joint sup-t bands, and `controls` are now supported — see the dedicated entries elsewhere in `[Unreleased]`). Results expose `results.path_effects: Dict[Tuple[int, ...], Dict[str, Any]]` and `results.to_dataframe(level="by_path")`; the summary grows a "Treatment-Path Disaggregation" block. Ties in path frequency are broken lexicographically on the path tuple for deterministic ranking. Overflow (`by_path > n_observed_paths`) returns all observed paths with a `UserWarning`. See `docs/methodology/REGISTRY.md` §ChaisemartinDHaultfoeuille `Note (Phase 3 by_path per-path event-study disaggregation)` for the full contract.
- **`ChaisemartinDHaultfoeuille.by_path` + `n_bootstrap > 0`** — bootstrap SE for per-path event-study effects. The top-k paths are enumerated once on the observed data (R-faithful path-stability semantics: matches `did_multiplegt_dyn(..., by_path=k, bootstrap=B)`, confirmed empirically against `DIDmultiplegtDYN 2.3.3`), and the existing multiplier bootstrap (`bootstrap_weights ∈ {"rademacher", "mammen", "webb"}`) runs per `(path, horizon)` target via the shared `_bootstrap_one_target` / `compute_effect_bootstrap_stats` helpers. Point estimates are unchanged from the analytical path. Bootstrap SE replaces the analytical SE in `path_effects[path]["horizons"][l]["se"]`, and `p_value` / `conf_int` propagate the **bootstrap percentile** statistics (library Round-10 convention, same as `overall` / `joiners` / `leavers` / `multi_horizon`); `t_stat` is SE-derived via `safe_inference` per the anti-pattern rule. Interpretation is *conditional on the observed path set* — practitioners wanting unconditional inference capturing path-selection uncertainty need a pairs-bootstrap (no R precedent). **SE inherits the analytical cross-path cohort-sharing deviation:** bootstrap input is the same full-panel cohort-centered path IF as the analytical path, so the bootstrap SE is a Monte Carlo analog of the analytical SE and inherits the existing analytical-path divergence from R on mixed-path cohorts (see REGISTRY.md for the full mechanism). On single-path-cohort panels, bootstrap and analytical SE both track R up to the Phase 2 envelope. **Deviation from R (CI method):** R's per-path bootstrap CI is normal-theory around the bootstrap SE (half-width ≈ `1.96·se`); ours is the bootstrap percentile CI, intentionally diverging from R to keep the dCDH inference surface internally consistent across all bootstrap targets. Positive regressions at `tests/test_chaisemartin_dhaultfoeuille.py::TestByPathBootstrap` (`@pytest.mark.slow`): point-estimate invariance, finite SE on non-degenerate panels, bootstrap-vs-analytical SE within 30% rtol on cohort-clean panels, degenerate-cohort NaN propagation, Rademacher/Mammen/Webb parity, seed reproducibility, and percentile-vs-normal-theory CI pinning. See `docs/methodology/REGISTRY.md` §ChaisemartinDHaultfoeuille `Note (Phase 3 by_path ...)` → **Bootstrap SE** for the full write-up.
- **R-parity for `ChaisemartinDHaultfoeuille.by_path`** against `DIDmultiplegtDYN 2.3.3`. Two new scenarios in `benchmarks/data/dcdh_dynr_golden_values.json` generated from `did_multiplegt_dyn(..., by_path=k)`: `mixed_single_switch_by_path` (2 paths, `by_path=2`) and `multi_path_reversible_by_path` (4 observed paths, `by_path=3`, via a new deterministic multi-path DGP pattern in the R generator). Per-path point estimates and per-path switcher counts match R exactly; per-path SE matches within the Phase 2 multi-horizon SE envelope (observed rtol ≤ 10.2% on the 2-path scenario, ≤ 4.2% on the 4-path scenario). Parity tests live at `tests/test_chaisemartin_dhaultfoeuille_parity.py::TestDCDHDynRParityByPath`, matching paths by tuple label via set-equality (robust to R's undocumented frequency-tie tiebreak) and cross-checking per-path switcher counts before SE comparison. **Deviation documented:** cross-path cohort sharing — our full-panel cohort-centered plug-in vs R's per-path re-run diverges materially when a `(D_{g,1}, F_g, S_g)` cohort spans multiple observed paths; the two coincide when every cohort is single-path. The parity scenarios are constructed to keep cohorts single-path (scenario 13 by design, scenario 14 via path-assignment-deterministic-on-F_g). See `docs/methodology/REGISTRY.md` §ChaisemartinDHaultfoeuille `Note (Phase 3 by_path...)` for the full write-up.
- **`profile_panel()` utility + `llms-autonomous.txt` reference guide (agent-facing)** — new `diff_diff.profile_panel(df, *, unit, time, treatment, outcome)` returns a frozen `PanelProfile` dataclass of structural facts (panel balance, treatment-type classification — `"binary_absorbing"` / `"binary_non_absorbing"` / `"continuous"` / `"categorical"`, cohort structure, outcome characteristics, and a `tuple[Alert, ...]` of factual observations). `.to_dict()` returns a JSON-serializable view. Paired with a new bundled `"autonomous"` variant on `get_llm_guide()` — `get_llm_guide("autonomous")` returns a reference-shaped guide (distinct from the existing workflow-prose `"practitioner"` variant) with §1 audience disclaimer, §2 `PanelProfile` field reference, §3 embedded 17-estimator × 9-design-feature support matrix, §4 per-design-feature reasoning citing Baker et al. (2025) and Roth / Sant'Anna (2023), §5 post-fit validation index, §6 BR/DR schema reference, §7 citations, §8 intentional omissions. Both pieces are bundled inside the wheel (no GitHub / RTD dependency at runtime); `diff_diff/__init__.py` module docstring leads with an agent-entry block listing `profile_panel`, `get_llm_guide("autonomous")`, `get_llm_guide("practitioner")`, and `BusinessReport` so `help(diff_diff)` surfaces them. Descriptive, not opinionated — `profile_panel` alerts never recommend a specific estimator, and the guide enumerates trade-offs rather than dispatching. Exports: `profile_panel`, `PanelProfile`, `Alert` from top-level `diff_diff`.
- **`target_parameter` block in BR/DR schemas (experimental; schema version bumped to 2.0)** — `BUSINESS_REPORT_SCHEMA_VERSION` and `DIAGNOSTIC_REPORT_SCHEMA_VERSION` bumped from `"1.0"` to `"2.0"` because the new `"no_scalar_by_design"` value on the `headline.status` / `headline_metric.status` enum (dCDH `trends_linear=True, L_max>=2` configuration) is a breaking change per the REPORTING.md stability policy. BusinessReport and DiagnosticReport now emit a top-level `target_parameter` block naming what the headline scalar actually represents for each of the 16 result classes. Closes BR/DR foundation gap #6 (target-parameter clarity). Fields: `name`, `definition`, `aggregation` (machine-readable dispatch tag), `headline_attribute` (raw result attribute), `reference` (citation pointer). BR's summary emits the short `name` right after the headline; DR's overall-interpretation paragraph does the same; both full reports carry a "## Target Parameter" section with the full definition. Per-estimator dispatch is sourced from REGISTRY.md and lives in the new `diff_diff/_reporting_helpers.py::describe_target_parameter`. A few branches read fit-time config (`EfficientDiDResults.pt_assumption`, `StackedDiDResults.clean_control`, `ChaisemartinDHaultfoeuilleResults.L_max` / `covariate_residuals` / `linear_trends_effects`); others emit a fixed tag (the fit-time `aggregate` kwarg on CS / Imputation / TwoStage / Wooldridge does not change the `overall_att` scalar — disambiguating horizon / group tables is tracked under gap #9). See `docs/methodology/REPORTING.md` "Target parameter" section.
- SyntheticDiD coverage Monte Carlo calibration table added to `docs/methodology/REGISTRY.md` §SyntheticDiD — rejection rates at α ∈ {0.01, 0.05, 0.10} across `placebo` / `bootstrap` / `jackknife` on 3 representative DGPs (balanced / exchangeable, unbalanced, and Arkhangelsky et al. (2021) AER §6.3 non-exchangeable). Artifact at `benchmarks/data/sdid_coverage.json` (500 seeds × B=200), regenerable via `benchmarks/python/coverage_sdid.py`.

### Fixed
- **SyntheticDiD `variance_method="bootstrap"` now runs the paper-faithful refit bootstrap** with R-default warm-start. Re-estimates ω̂_b and λ̂_b via two-pass sparsified Frank-Wolfe on each pairs-bootstrap draw using the fit-time normalized-scale zeta — Arkhangelsky et al. (2021) Algorithm 2 step 2, matching the behavior of R's default `synthdid::vcov(method="bootstrap")` (which rebinds `attr(estimate, "opts")` so the renormalized ω serves as Frank-Wolfe initialization). The Python path threads that warm-start through `compute_sdid_unit_weights(..., init_weights=_sum_normalize(ω̂[boot_control_idx]))` and `compute_time_weights(..., init_weights=λ̂)` on each bootstrap draw. `compute_sdid_unit_weights` and `compute_time_weights` gain a new `init_weights` kwarg; when provided, the Rust top-level fast-path is skipped in favor of the Python two-pass dispatcher (whose inner FW calls still dispatch to Rust). Without this kwarg both helpers remain backward-compatible and keep the Rust fast-path. The previous fixed-weight bootstrap path is removed entirely — it was not paper-faithful and, despite prior documentation claiming otherwise, also did not match R's default bootstrap (the previous R-parity test fixture invoked `synthdid_estimate(weights=...)` without rebinding `opts`, which silently runs fixed-weight, so the 1e-10 parity was between two paths both wrong in the same direction). Coverage MC at the new artifact above quantifies the correctness fix on 3 representative null DGPs. **Users' existing `variance_method="bootstrap"` fits will return materially different SE / p-value / CI values on the next release** — same enum name, corrected semantics. Bootstrap is now ~5–30× slower per fit than the old fixed-weight shortcut (panel-size dependent; warm-start converges faster than cold-start so the slowdown is less than the 10–100× prior estimate). The PR #349 follow-on bullets below (analytical p-value dispatch, sqrt((r-1)/r) SE formula, retry-to-B contract) all carry over to the refit path unchanged.
- SyntheticDiD `variance_method="bootstrap"` now computes p-values from the analytical normal-theory formula using the bootstrap SE (matching R's `synthdid::vcov()` convention), rather than an empirical null-distribution formula that is not valid for bootstrap draws. `is_significant` and `significance_stars` are derived from `p_value` and will also change for bootstrap fits. Placebo and jackknife are unchanged. Point estimates are unaffected.
- SyntheticDiD bootstrap SE formula applies the `sqrt((r-1)/r)` correction matching R's synthdid and the placebo SE formula.
- SyntheticDiD bootstrap now retries degenerate resamples (all-control or all-treated, or non-finite `τ_b`) until exactly `n_bootstrap` valid replicates are accumulated, matching R's `synthdid::bootstrap_sample` and Arkhangelsky et al. (2021) Algorithm 2. Previously the Python path counted attempts (with degenerate draws silently dropped), producing fewer valid replicates than requested. A bounded-attempt guard (`20 × n_bootstrap`) prevents pathological-input hangs.
- **TROP global bootstrap SE backend parity under fixed seed** — Rust and Python backends now produce bit-identical bootstrap SE under the same `seed`. Previously Rust's `bootstrap_trop_variance_global` seeded `rand_xoshiro::Xoshiro256PlusPlus` per replicate while Python's fallback consumed `numpy.random.default_rng` (PCG64), producing ~28% SE divergence on tiny panels under `seed=42`. Fixed by extracting a shared `stratified_bootstrap_indices` helper in `diff_diff/bootstrap_utils.py` that pre-generates per-replicate stratified sample indices via numpy on the Python side; both backends consume the same integer arrays through the PyO3 surface. Sampling law (stratified: controls then treated, with replacement) is unchanged. Closes the bootstrap-RNG half of silent-failures audit finding #23 (grid-search half closed in PR #348; local-method methodology half closed by the two Fixed entries below). Local-method TROP also adopts the Python-canonical index contract for the RNG layer here.
- **TROP local-method Rust weight-matrix no longer normalized** — `rust/src/trop.rs::compute_weight_matrix` no longer divides time-weights or unit-weights by their respective sums before the outer product. The paper's Equation 2/3 (Athey, Imbens, Qu, Viviano 2025) and REGISTRY.md Requirements checklist (line 2037: `[x] Unit weights: exp(-λ_unit × distance) (unnormalized, matching Eq. 2)`) both specify raw-exponential weights; Python's `_compute_observation_weights` was already REGISTRY-compliant. **User-visible effect**: Rust local-method ATT values may shift for any fit with `lambda_nn < infinity` — normalizing the weight-matrix inflated the effective nuclear-norm penalty relative to the data-fit term, changing the regularization trade-off. For `lambda_nn = infinity` (factor model disabled) outputs are unchanged because uniform weight scaling leaves the minimum-norm WLS argmin invariant. Rust LOOCV-selected lambdas may also shift on this boundary; both backends now converge on the same REGISTRY-compliant selection.
- **TROP local-method Python `_compute_observation_weights` now uses the function-argument `Y, D` and treats all non-target units as donors** — two coupled changes that bring Python structurally in line with Rust and the paper's Eq. 2/3:
    1. Removed the `if self._precomputed is not None:` branch that silently substituted `self._precomputed["Y"]` / `["D"]` / `["time_dist_matrix"]` (original-panel cache populated during main fit) for the function-argument `Y, D`. Under bootstrap, `_fit_with_fixed_lambda` computes fresh `Y, D` from the resampled `boot_data` and passes them in; the helper was discarding those and recomputing unit distances from the original panel, so Python's local bootstrap resampled units but reused stale unit-distance weights. Rust's bootstrap was already correct (always consumed `y_boot, d_boot`).
    2. Removed the `valid_control_at_t = D[t, :] == 0` target-period donor gate that zeroed `ω_j` for any unit `j` treated at the target period (other than the target unit itself). Per REGISTRY Eq. 2/3 and Rust's `compute_weight_matrix`, `ω_j = exp(-λ_unit × dist(j, i))` for all `j ≠ i`; treated-cell exclusion happens via the `(1 − W_{js})` factor applied inside `_estimate_model`. Same-cohort donors now contribute via their pre-treatment rows. Empirically the main-fit ATT is unchanged on tested fixtures because same-cohort pre-treatment observations are exactly absorbed by their own unit fixed effect `alpha_j` without propagating into `mu`, `beta`, or other units' parameters — so this change is structural alignment rather than a numerical shift in output. Users on same-cohort panels with very few controls may still see tiny differences in edge cases; the new `test_local_method_same_cohort_donor_parity` regression guards the aligned behavior.
  Together with the normalization fix above, TROP local-method backend parity on the main-fit ATT is regime-dependent: `atol=rtol=1e-14` for `lambda_nn=inf` (no nuclear-norm regularization, uniform weight scaling leaves the WLS argmin invariant) and `atol=1e-10` for finite `lambda_nn` (FISTA inner loop + BLAS reduction ordering introduce sub-1e-10 roundoff across Rust `faer` vs numpy paths). Bootstrap SE parity is asserted at `atol=1e-5` to accommodate ~1e-7 roundoff between Rust's `estimate_model` matrix factorization and numpy's `lstsq` that accumulates across per-replicate fits; sub-1e-14 bootstrap parity is tracked as a follow-up in `TODO.md` under "unify Rust local-method solver path". Closes silent-failures audit finding #23 (local-method half; the RNG half closed in PR #354 and the grid-search half in PR #348).

### Changed
- **`did_had_pretest_workflow(aggregate="event_study")` verdict no longer emits the "paper step 2 deferred to Phase 3 follow-up" caveat** — the joint pre-trends Stute test closes that gap. The two-period `aggregate="overall"` path retains the existing caveat since the joint variant does not apply to single-pre-period panels. Downstream code that greps verdict strings for the Phase 3 caveat will see it suppressed on the event-study path.
- **SyntheticDiD bootstrap no longer supports survey designs** (capability regression in PR #351, **restored in PR #355** — see Added/Changed entries directly below). The removed fixed-weight bootstrap path was the only SDID variance method that supported strata/PSU/FPC (via Rao-Wu rescaled bootstrap); the PR #351 paper-faithful refit bootstrap initially rejected all survey designs (including pweight-only) with `NotImplementedError`. PR #355 restores the capability via a weighted-FW + Rao-Wu composition; the lock-out window applies only to the v3.2.x line that ships PR #351 alone (without PR #355). Composing Rao-Wu rescaled weights with Frank-Wolfe re-estimation: see `docs/methodology/REGISTRY.md` §SyntheticDiD `Note (survey + bootstrap composition)`.

### Added (PR #355)
- **SDID `variance_method="bootstrap"` survey support restored** via a hybrid pairs-bootstrap + Rao-Wu rescaling composed with a weighted Frank-Wolfe kernel. Each bootstrap draw first performs the unit-level pairs-bootstrap resampling specified by Arkhangelsky et al. (2021) Algorithm 2 (`boot_idx = rng.choice(n_total)`), and *then* applies Rao-Wu rescaled per-unit weights (Rao & Wu 1988) sliced over the resampled units — NOT a standalone Rao-Wu bootstrap. New Rust kernel `sc_weight_fw_weighted` (and `_with_convergence` sibling) accepts a per-coordinate `reg_weights` argument so the FW objective becomes `min ||A·ω - b||² + ζ²·Σ_j reg_w[j]·ω[j]²`. New Python helpers `compute_sdid_unit_weights_survey` and `compute_time_weights_survey` thread per-control survey weights through the two-pass sparsify-refit dispatcher (column-scaling Y by `rw` for the loss, `reg_weights=rw` for the penalty on the unit-weights side; weighted column-centering + row-scaling Y by `sqrt(rw)` for the loss with uniform reg on the time-weights side). `_bootstrap_se` survey branch composes the per-draw `rw` (Rao-Wu rescaling for full designs, constant `w_control` for pweight-only fits) with the weighted-FW helpers, then composes `ω_eff = rw·ω/Σ(rw·ω)` for the SDID estimator. Coverage MC artifact extended with a `stratified_survey` DGP (BRFSS-style: N=40, strata=2, PSU=2/stratum); the bootstrap row's near-nominal calibration is the validation gate (target rejection ∈ [0.02, 0.10] at α=0.05). New regression tests across `test_methodology_sdid.py::TestBootstrapSE` (single-PSU short-circuit, full-design and pweight-only succeeds-tests, zero-treated-mass retry, deterministic Rao-Wu × boot_idx slice) and `test_survey_phase5.py::TestSyntheticDiDSurvey` (full-design ↔ pweight-only SE differs assertion). See REGISTRY.md §SyntheticDiD ``Note (survey + bootstrap composition)`` for the full objective and the argmin-set caveat.

### Changed (PR #355)
- **SDID bootstrap SE values under survey fits now differ numerically from the v3.2.x line that shipped PR #351 alone**: the fit no longer raises `NotImplementedError`, and instead returns the weighted-FW + Rao-Wu SE. Non-survey fits are unaffected (the bootstrap dispatcher routes only the survey branch through the new `_survey` helpers; non-survey fits continue to call the existing `compute_sdid_unit_weights` / `compute_time_weights` and stay bit-identical at rel=1e-14 on the `_BASELINE["bootstrap"]` regression). SDID's `placebo` and `jackknife` paths still reject `strata/PSU/FPC` on the v3.2.x line; full-design support for those methods lands separately in the entries below.

### Added
- **SDID `variance_method="placebo"` and `"jackknife"` now support strata/PSU/FPC designs.** Closes the last SDID survey gap. All three variance methods (bootstrap from PR #355, plus placebo and jackknife here) now handle full survey designs. New private methods `SyntheticDiD._placebo_variance_se_survey` and `_jackknife_se_survey` route the full-design path through method-specific allocators:
  - **Placebo** — stratified permutation (Pesarin 2001). Each draw samples pseudo-treated indices uniformly without replacement from controls *within each stratum* containing actual treated units; non-treated strata contribute their controls unconditionally. The weighted Frank-Wolfe kernel from PR #355 (`compute_sdid_unit_weights_survey` / `compute_time_weights_survey`) re-estimates ω and λ per draw with per-control survey weights threaded into both loss and regularization; post-optimization composition `ω_eff = rw·ω/Σ(rw·ω)`. Arkhangelsky Algorithm 4 SE formula unchanged.
  - **Jackknife** — PSU-level leave-one-out with stratum aggregation (Rust & Rao 1996). `SE² = Σ_h (1-f_h)·(n_h-1)/n_h·Σ_{j∈h}(τ̂_{(h,j)} - τ̄_h)²` with `f_h = n_h_sampled / fpc[h]` (population-count FPC form). λ held fixed across LOOs; ω subsetted, composed with rw, renormalized. Strata with `n_h < 2` silently skipped (matches R `survey::svyjkn` with `lonely_psu="remove"` / `"certainty"`; `"adjust"` raises `NotImplementedError`). Full-census strata (`f_h ≥ 1`) short-circuit to zero contribution before any LOO feasibility check. `SE = 0` is returned for legitimate zero variance (e.g., every stratum full-census); `SE = NaN` with a targeted `UserWarning` is reserved for undefined cases — all strata skipped, or any delete-one replicate in a non-full-census contributing stratum is undefined (all-treated-in-one-PSU LOO, kept ω_eff / w_treated mass zero, estimator raises). Unstratified single-PSU short-circuits to NaN.
  - **Fit-time feasibility guards** (placebo): `ValueError` on stratum-level infeasibility with targeted messages distinguishing three cases — **Case B** (treated-containing stratum has zero controls), **Case C** (fewer controls than treated in a treated stratum), **Case D** (every treated stratum is exact-count `n_c_h == n_t_h` → permutation support is 1, null distribution collapses). Partial-permutation fallback rejected because it would silently change the null-distribution semantics.
  - **Gate relaxed**: the fit-time guard at `synthetic_did.py:352-369` that rejected placebo/jackknife + strata/PSU/FPC is removed. Replicate-weight designs remain rejected (separate methodology — replicate variance is closed-form and would double-count with Rao-Wu-like rescaling). Non-survey and pweight-only paths bit-identical by construction — the new code is gated on `resolved_survey_unit.(strata|psu|fpc) is not None`.
  - **Coverage MC**: `benchmarks/data/sdid_coverage.json` extended with jackknife on `stratified_survey`. Bootstrap validates near-nominal (α=0.05 rejection = 0.058, SE/trueSD = 1.13). Jackknife reported with an anti-conservatism caveat: with only 2 PSUs per stratum the stratified jackknife formula has 1 effective DoF per stratum, a well-documented limitation of Rust & Rao (1996) — `se_over_truesd ≈ 0.46` on this DGP. Users needing tight SE calibration with few PSUs should prefer `variance_method="bootstrap"`. Placebo is structurally infeasible on the existing `stratified_survey` DGP (its cohort packs into one stratum with 0 never-treated units — by design a bootstrap-suited DGP); the placebo survey path is exercised via unit tests on a feasible fixture.
  - **Regression tests** across `tests/test_survey_phase5.py`: two new classes `TestSDIDSurveyPlaceboFullDesign` and `TestSDIDSurveyJackknifeFullDesign`. Placebo: pseudo-treated-stratum contract, Case B / Case C front-door guards with targeted-message regression, SE-differs-from-pweight-only, deterministic dispatch. Jackknife: stratum-aggregation self-consistency, **FPC magnitude regression** (2-stratum handcrafted panel asserts `SE_fpc == SE_nofpc · sqrt(1-f)` at `rtol=1e-10`), single-PSU-stratum skip, unstratified short-circuit, all-strata-skipped warning + NaN, SE-differs-from-pweight-only, deterministic dispatch. Existing `test_full_design_placebo_raises` and `test_full_design_jackknife_raises` flipped to `_succeeds` assertions. All 19 existing pweight-only and non-survey placebo/jackknife tests pass unchanged (bit-identity preserved via the new-path gating).
  - **Allocator asymmetry** (documented in REGISTRY): placebo ignores the PSU axis (unit-level within-stratum permutation — the classical stratified permutation test; PSU-level permutation on few PSUs is near-degenerate); jackknife respects PSU (PSU-level LOO is the canonical survey jackknife). Both respect strata. See `docs/methodology/REGISTRY.md` §SyntheticDiD `Note (survey + placebo composition)` and `Note (survey + jackknife composition)`.

## [3.2.0] - 2026-04-19

### Added
- **`BusinessReport` and `DiagnosticReport` (experimental preview)** (PR #318) - practitioner-ready output layer. `BusinessReport(results, ...)` produces plain-English narrative summaries (`.summary()`, `.full_report()`, `.export_markdown()`, `.to_dict()`) from any of the 16 fitted result types. `DiagnosticReport(results, ...)` orchestrates the existing diagnostic battery (parallel trends, pre-trends power, HonestDiD sensitivity, Goodman-Bacon, heterogeneity, design-effect, EPV) plus estimator-native diagnostics for SyntheticDiD (`pre_treatment_fit`, weight concentration, in-time placebo, zeta sensitivity) and TROP (factor-model fit metrics). Both classes expose an AI-legible `to_dict()` schema (single source of truth; prose renders from the dict). BR auto-constructs DR by default so summaries mention pre-trends, robustness, and design-effect findings in one call. See `docs/methodology/REPORTING.md` for methodology deviations including the no-traffic-light-gates decision, pre-trends verdict thresholds (0.05 / 0.30), and power-aware phrasing driven by `compute_pretrends_power`. **Both schemas are marked experimental in this release** - wording, verdict thresholds, and schema shape will change; do not anchor downstream tooling on them yet.
- **Kernel / local-linear / nonparametric infrastructure** (PRs #327, #335) - bandwidth selector, local linear regression, HC2 / Bell-McCaffrey variance helpers, and a port of R `nprobust`'s point-estimate path. Foundation for the upcoming `HeterogeneousAdoptionDiD` estimator (de Chaisemartin, Ciccia, D'Haultfœuille & Knau 2024 — "DiD with no untreated group"). Released as internal modules with full test coverage (`tests/test_bandwidth_selector.py`, `tests/test_local_linear.py`, `tests/test_linalg_hc2_bm.py`, `tests/test_nprobust_port.py`); the user-facing estimator ships in a later phase.
- **Cell-period IF allocator for dCDH survey variance (Class A contract)** (PR #323) - replaces the group-level allocator `ψ_i = ψ_g * (w_i / W_g)` with a cell-period allocator `ψ_i = ψ_g * (w_i / W_{g, out_idx})` on the post-period cell for the DID_l replicate-weight ATT path. Is the allocator shape that the v3.2.0 heterogeneity and bootstrap extensions below build on. Documents the post-period attribution convention in REGISTRY.md with a hand-computed row-sum identity test.

### Performance
- **`aggregate_survey` stratum-PSU scaffolding precompute** — the per-cell Taylor-series variance inside `aggregate_survey` no longer rebuilds stratum-PSU scaffolding on every cell. A frozen `_PsuScaffolding` (strata codes, global PSU codes unique across strata, per-stratum counts and FPC ratios, singleton mask, static legitimate-zero counts and variance-computable flag) is precomputed once per design at the top of `aggregate_survey` and threaded through `_cell_mean_variance` to a new `_compute_if_variance_fast` path that replaces the per-stratum pandas groupby with two vectorized `np.bincount` passes. BRFSS-shaped 50-state × 10-year × 1M-row microdata → state-year panel drops from ~24s to sub-2s under both backends (the path is pure Python, so Python and Rust track each other). Numerical output is preserved to sub-ULP tolerance; seven-case equivalence tests (`TestAggregateSurveyScaffolding`) assert `assert_allclose(atol=1e-14, rtol=1e-14)` between fast and legacy paths across stratified+PSU+FPC, stratified no FPC, PSU-only, weights-only, and all three `lonely_psu` modes (remove / certainty / adjust). Replicate-weight designs continue to route through `compute_replicate_if_variance` unchanged. `_compute_stratified_psu_meat` is untouched — all other TSL callers (DiD / TWFE / CS / etc.) are unaffected.

### Changed
- Add Zenodo DOI badge to README; upgrade the BibTeX citation block with the concept DOI (`10.5281/zenodo.19646175`) and list author as Isaac Gerber (matching `CITATION.cff`). `CITATION.cff` carries the concept DOI as its top-level `doi:` field — Zenodo auto-mints a versioned DOI for every release, but the CFF file tracks the concept DOI only so it doesn't need a follow-up edit per release. DOI was minted by Zenodo when v3.1.3 was released.
- **`ChaisemartinDHaultfoeuille` heterogeneity + within-group-varying PSU/strata now supported under Binder TSL** - `fit(heterogeneity=..., survey_design=...)` no longer raises `NotImplementedError` when the resolved design's PSU or strata vary across the cells of a group. On the **Binder TSL** branch (`compute_survey_if_variance`), the heterogeneity WLS coefficient IF is expanded to observation level via the cell-period allocator `ψ_i = ψ_g * (w_i / W_{g, out_idx})` on the post-period cell — the DID_l post-period single-cell convention shipped in v3.1.x. Under PSU=group the PSU-level Binder TSL variance is byte-identical to the previous release (PSU-level aggregate telescopes to `ψ_g`); under within-group-varying PSU, mass lands in the post-period PSU of the transition. The **Rao-Wu replicate-weight** branch (`compute_replicate_if_variance`) retains the legacy group-level allocator `ψ_i = ψ_g * (w_i / W_g)`: replicate variance computes `θ_r = sum_i ratio_ir * ψ_i` at observation level and is therefore not PSU-telescoping, so the cell-period allocator would silently change the replicate SE whenever a replicate column's ratios vary within group (e.g., per-row replicate matrices). Replicate + heterogeneity fits therefore produce byte-identical SE to the previous release, and the newly-unblocked `heterogeneity=` + within-group-varying PSU combination is unreachable under replicate designs by construction (`SurveyDesign` rejects `replicate_weights` combined with explicit `strata/psu/fpc`).
- **`ChaisemartinDHaultfoeuille.fit(survey_design=..., n_bootstrap > 0)` now supports within-group-varying PSU** — the PSU-level Hall-Mammen wild multiplier bootstrap has been extended from a group-level PSU map (one multiplier per group) to a cell-level PSU map (one multiplier per `(g, t)` cell's PSU). A dispatcher in `_compute_dcdh_bootstrap` detects PSU-within-group-constant regimes (including PSU=group auto-inject and strictly-coarser PSU with within-group constancy) and routes them through the legacy group-level path so the bootstrap SE is bit-identical to the previous release (guarded by the new `test_bootstrap_se_matches_pre_pr4_baseline` and the pre-existing `test_auto_inject_bit_identical_to_group_level`). Under within-group-varying PSU, a group contributing cells to multiple PSUs receives independent multiplier draws per PSU — the correct Hall-Mammen wild PSU clustering at cell granularity. Multi-horizon bootstraps draw a single shared `(n_bootstrap, n_psu)` PSU-level weight matrix per block and broadcast per-horizon via each horizon's cell-to-PSU map, so the sup-t simultaneous confidence band remains a valid joint distribution. Closes the last `NotImplementedError` gate in the dCDH survey contract; replicate-weight variance and `n_bootstrap > 0` remain mutually exclusive by construction. **Scope note:** panels with *terminal missingness* where the terminally-missing group is in a cohort whose other groups still contribute at the missing period now raise a targeted `ValueError` on every survey variance path that uses the cell-period allocator: Binder TSL with within-group-varying PSU, Rao-Wu replicate-weight ATT (which always uses the cell allocator per the Class A contract shipped in PR #323), and the cell-level wild PSU bootstrap. Cohort-recentering leaks centered IF mass onto cells with no positive-weight observations, which the cell-period allocator cannot attach to any observation/PSU. This closes a silent mass-drop bug the cell-period allocator introduced across all three paths in v3.1.x; pre-process the panel to remove terminal missingness (drop late-exit groups or trim to a balanced sub-panel) as the documented workaround. For Binder TSL only, using an explicit `psu=<group_col>` routes through the legacy group-level allocator where the row-sum identity makes the two allocators statistically equivalent. Replicate-weight ATT and within-group-varying-PSU bootstrap have no such allocator fallback — the panel itself must be pre-processed. PSU-within-group-constant Binder TSL (including PSU=group auto-inject) is unaffected.
- **Performance review: practitioner-scale scenarios + benchmark harness extension** (PR #333) - new `docs/performance-scenarios.md` documents 5-7 realistic practitioner workflows (marketing lift, geo-experiment, BRFSS state-policy, dCDH reversible treatment) grounded in the practitioner docs and the paper literature, not cookie-cutter textbook data. `benchmarks/speed_review/` extended with practitioner-scale scripts and per-backend bit-identity baselines. Baselines refreshed against current main. Finding: the biggest leverage areas are bootstrap resampling loops and per-replicate survey-design rebuilds in the bootstrap path; documented in `docs/performance-plan.md` for follow-up optimization PRs.
- **Wall-clock timing tests excluded from default CI** (PRs #330, #336) - `TestCallawaySantAnnaSEAccuracy.test_timing_performance` and `TestPerformanceRegression` marked `@pytest.mark.slow`, removing false-positive CI failures from runner-noise variance (BLAS path variation, neighbor VM contention). Tests remain runnable via `pytest -m slow` for ad-hoc local benchmarking; the perf-review harness above is the principled replacement for CI-gated performance tracking.

### Fixed
- **Silent-failures audit: axis A** (PR #334) — minor solver paths numerical-precision / scale-fragility closeouts, completing the SDID extreme-Y-scale work started in v3.1.2.
- **Silent-failures audit: axis C & J** (PR #339) — B-spline derivative warning scope broadened; `SurveyPowerConfig` stale-cache wording narrowed.
- **Silent-failures audit: axis E** (PR #331) — row-drop counters surfaced across estimator paths so silent validator row-drops leave an explicit count on the result.
- **Silent-failures audit: axis G** (PR #337) — Rust vs Python backend edge-case parity tests added for rank-deficient, extreme-scale, and constant-column inputs.
- **SyntheticDiD diagnostic Y-normalization parity** (PR #328) — extends the PR #312 catastrophic-cancellation fix from the main fit path into `SyntheticDiDResults.in_time_placebo()` and `.sensitivity_to_zeta_omega()`. Diagnostics now apply the same `Y_shift / Y_scale` normalization the main fit uses, pass `zeta / Y_scale` and a normalized `min_decrease` into Frank-Wolfe, then rescale `att` / `pre_fit_rmse` back to original-Y units.
- **TROP bootstrap failure-rate guards** (PR #324) — alternating-minimization bootstrap loops now emit a `UserWarning` on silent high-failure-rate runs (LOOCV and bootstrap aggregation paths both covered); attempt-count-based warning replaces the previous observation-count denominator that could silently mask sparse runs.
- **`simulate_power()` failure-count surface + narrow except clause** (PR #326) — power-simulation replicate loop narrows the exception whitelist from `except Exception` to estimation/data-path failures (`TypeError` and friends now propagate, not silently absorb), and surfaces `n_simulation_failures` on `SimulationPowerResults`. Failure count included in `summary()` and `to_dict()`.

## [3.1.3] - 2026-04-18

### Added
- **Replicate-weight variance and PSU-level bootstrap for dCDH** (PR #311) - `ChaisemartinDHaultfoeuille` now accepts `variance_method="replicate"` for BRR / Fay / JK1 / JKn / SDR inference, and PSU-level multiplier bootstrap when `survey_design.psu` is set. Adds df-aware inference (reduced effective df under replicate variance; propagated through delta / HonestDiD surfaces) plus group-level PSU map construction. Validated via per-cohort aggregation, shared-draw multi-horizon bootstrap alignment, and cross-surface df consistency.
- **Zenodo DOI auto-minting configuration** (PR #321) - `.zenodo.json` at repo root defines release metadata so the next GitHub Release automatically mints a Zenodo DOI (concept DOI + versioned DOI). Also adds a top-level `LICENSE` file for Zenodo archival.

### Fixed
- **Silent sparse→dense lstsq fallback in `ImputationDiD` and `TwoStageDiD`** (PR #319) - when the sparse solver fails and the dense fallback runs, the estimator now emits a `UserWarning` instead of silently switching paths. Regression tests assert the dense fallback SEs remain usable.
- **Non-convergence signaling in TROP alternating-minimization solvers** (PR #317) - the global- and local-TROP solvers now emit a `UserWarning` when the alternating-minimization loop exits without meeting tolerance, including LOOCV and bootstrap aggregation paths. Warnings aggregate at top-level call sites to avoid log spam.

### Changed
- **`/bump-version` skill updates `CITATION.cff`** (PR #320) - internal release-management tooling now keeps `CITATION.cff` `version:` and `date-released:` in sync with the other version surfaces. Resolves a single `RELEASE_DATE` upfront (from the CHANGELOG header if pre-populated, else today's date) and threads it through all date-bearing files — fixes drift that caused v3.1.2 to ship with `CITATION.cff` still pinned at 3.1.1.

## [3.1.2] - 2026-04-18

### Fixed
- **SyntheticDiD catastrophic cancellation at extreme Y scale** (PR #312) - the Frank-Wolfe weight solver lost precision when outcome magnitudes were very large or very small; results are now numerically stable across scales.
- **Non-convergence signaling in FE imputation alternating-projection solvers** (PR #314) - `ImputationDiD`, `TwoStageDiD`, and shared `within_transform` now emit a `UserWarning` when the alternating-projection / weighted-demean loop exits without meeting the tolerance. `max_iter` and `tol` are documented on `within_transform`.
- **Non-convergence signaling in SyntheticDiD Frank-Wolfe solver** (PR #315) - the numpy-path Frank-Wolfe SC weight solver now emits a `UserWarning` when the loop exits without meeting `min_decrease`. Wrapper-level and `max_iter=0` regression tests added.

### Changed
- Refresh `ROADMAP.md` to drop top-level phase numbering and reflect shipped state through v3.1.1 (PR #313). Absorbs dCDH into the Current State estimator list; adds Recently Shipped summary; reorganizes open work as Shipping Next / Under Consideration / AI-Agent Track / Long-term. Updates `docs/business-strategy.md`, `docs/survey-roadmap.md`, `docs/practitioner_decision_tree.rst`, `docs/choosing_estimator.rst`, `docs/api/chaisemartin_dhaultfoeuille.rst`, `README.md`, and `diff_diff/guides/llms-full.txt` to remove stale phase-deferral language now that the deferred items have shipped.
- Bump the `SyntheticDiD(lambda_reg=...)` and `SyntheticDiD(zeta=...)` deprecation warnings' removal target from `v3.1` to `v4.0.0`. Removing public kwargs in a patch / minor release would violate Semantic Versioning; the deprecation stays warning-only throughout the `3.x` line and will be removed in the next major release. Use `zeta_omega` / `zeta_lambda` instead.

## [3.1.1] - 2026-04-16

### Added
- **Jackknife variance estimation for SyntheticDiD** - `variance_method='jackknife'` implements the delete-one-unit jackknife from Arkhangelsky et al. (2021) Section 5. Supports both standard and survey-weighted jackknife with automatic `pweight` propagation. Validated against R `synthdid` package.
- **LinkedIn carousel** for dCDH estimator announcement (`carousel/diff-diff-dcdh-carousel.pdf`)

## [3.1.0] - 2026-04-14

### Added
- **dCDH Phase 3: Complete feature set for `ChaisemartinDHaultfoeuille`** - three sub-releases completing the estimator:
  - **Phase 3a** (PR #300): Placebo SE via multiplier bootstrap (resolves Phase 1 deferral), non-binary treatment support with crossing-cell detection and automatic cell dropping, R parity SE assertions tightened
  - **Phase 3b** (PR #302): Covariate adjustment via `controls` parameter (OLS residualization, Design 2 per-period path for non-binary treatment), group-specific linear trends via `trends_linear=True` (absorbs group-specific slopes before DiD), R `DIDmultiplegtDYN` parity tests for covariates and trends
  - **Phase 3c** (PR #303): HonestDiD sensitivity analysis integration - `honest_did()` method on results with automatic event-study-to-sensitivity bridge, support trimming for non-consecutive horizons, `l_vec` target specification, Delta-RM and Delta-SD smoothness bounds

### Changed
- ROADMAP.md updated: dCDH Phase 3 items marked shipped

## [3.0.2] - 2026-04-12

### Added
- **`ChaisemartinDHaultfoeuille`** (alias `DCDH`) - de Chaisemartin & D'Haultfœuille estimator for **non-absorbing (reversible) treatments**. The only modern staggered DiD estimator that handles treatment switching on AND off. Implements `DID_M` from AER 2020, validated against R `DIDmultiplegtDYN` v2.3.3. Ships Phases 1 and 2:
  - Phase 1: headline `DID_M` with analytical SE, joiners/leavers decompositions, single-lag placebo, multiplier bootstrap, TWFE decomposition diagnostic
  - Phase 2: multi-horizon event study (`L_max`), dynamic placebos, normalized estimator, cost-benefit aggregate (Lemma 4), sup-t simultaneous confidence bands, `plot_event_study()` integration
- **`twowayfeweights()`** - standalone TWFE decomposition diagnostic (Theorem 1, AER 2020)
- **`generate_reversible_did_data()`** - reversible-treatment panel data generator with 7 switch patterns
- **Survey-aware power analysis** - analytical helpers (`compute_power()`, `compute_mde()`, `compute_sample_size()`) accept a `deff` parameter for design-effect adjustment. Simulation helpers (`simulate_power`, `simulate_mde`, `simulate_sample_size`) accept a `survey_config` (`SurveyPowerConfig`) that generates data with complex survey structure and injects a `SurveyDesign` into each simulated fit.
- **`aggregate_survey()` `second_stage_weights` parameter** - choose `"pweight"` (default, population weights) or `"aweight"` (precision weights). pweight output is compatible with all survey-capable estimators; aweight is opt-in for GLS efficiency with estimators marked Full in the survey support matrix.
- **`conditional_pt` parameter** on `generate_survey_did_data()` - simulates scenarios where unconditional parallel trends fail but conditional PT holds after covariate adjustment
- **Tutorial 18: Geo-Experiment Analysis** (`18_geo_experiments.ipynb`) - SyntheticDiD walkthrough for marketing analytics: simulated DMA panel, 5 treated markets, fit + diagnostics + stakeholder summary
- **Practitioner decision tree** (`docs/practitioner_decision_tree.rst`) - "which method fits my business problem?" guide
- **Practitioner getting started guide** (`docs/practitioner_getting_started.rst`) - end-to-end walkthrough with terminology bridge
- **JOSS paper** (`paper.md`, `paper.bib`) - software paper for Journal of Open Source Software submission
- **CONTRIBUTORS.md** - author and contributor credit
- **Standalone CI Gate workflow** (`.github/workflows/ci-gate.yml`) - doc-only PRs no longer block on path-filtered test workflows

### Changed
- `aggregate_survey()` default second-stage weights changed from `aweight` (precision) to `pweight` (population). Users who need the old precision-weighting behavior can pass `second_stage_weights="aweight"`.
- README "For Data Scientists" section with practitioner-facing links and `aggregate_survey()` documentation
- CITATION.cff updated with version and release date
- ROADMAP.md updated: B1a-d marked done, B2b marked done, B3d marked shipped, dCDH entry updated with correct citations

### Fixed
- Doc-only PRs no longer block indefinitely on CI Gate (standalone gate workflow runs on all PRs regardless of path filters)
- `aggregate_survey()` docs no longer overclaim universal estimator compatibility - explicitly document aweight/pweight restrictions per the survey support matrix

## [3.0.1] - 2026-04-07

### Added
- **`aggregate_survey()`** — new function in `diff_diff.prep` that bridges individual-level survey microdata to geographic-period panels for DiD estimation. Computes design-based cell means and precision weights using domain estimation (Lumley 2004), with SRS fallback for small cells. Returns a panel DataFrame and pre-configured `SurveyDesign` for second-stage estimation. Supports both TSL and replicate-weight variance.
- **Python 3.14 support** — upgraded PyO3 from 0.22 to 0.28, updated CI and publish workflow matrices, bumped Rust MSRV to 1.84 for faer 0.24 compatibility.

### Changed
- Updated README Python support matrix to include 3.14

### Fixed
- Fix domain estimation zero-padding for correct design-based cell variance
- Fix SRS fallback weight normalization for scale invariance across replicate designs
- Validate numeric dtype for outcomes/covariates before aggregation (nullable dtype support)
- Validate grouping columns for NaN values

## [3.0.0] - 2026-04-07

v3.0 completes the survey support roadmap: all 16 estimators (15 inference-level +
BaconDecomposition diagnostic) now accept `survey_design`. See v2.8.0–v2.9.1 entries
for the full feature history leading to this release.

### Breaking Changes
- **Remove `bootstrap_weight_type` parameter** from CallawaySantAnna — use `bootstrap_weights` instead (deprecated since v1.0.1)
- **Remove TROP `method="twostep"` alias** — use `method="local"` (deprecated since v2.7.2)
- **Remove TROP `method="joint"` alias** — use `method="global"` (deprecated since v2.7.2)

### Upgrading from v2.x
- `CallawaySantAnna(bootstrap_weight_type="mammen")` → `CallawaySantAnna(bootstrap_weights="mammen")`
- `TROP(method="twostep")` → `TROP(method="local")`
- `TROP(method="joint")` → `TROP(method="global")`

### Deprecated
- SyntheticDiD `lambda_reg` and `zeta` parameters formally scheduled for removal in v3.1 — use `zeta_omega`/`zeta_lambda` instead

### Changed
- Internal attribute `bootstrap_weight_type` renamed to `bootstrap_weights` in bootstrap mixin and StaggeredTripleDifference for consistency
- TROP `set_params()` now validates `method` against `("local", "global")` — previously only validated in `__init__`
- Documentation updated: all survey gap notes for WooldridgeDiD removed, ROADMAP Phase 10 items marked shipped

## [2.9.1] - 2026-04-06

### Added
- **Survey theory document** (`docs/methodology/survey-theory.md`) — formal justification for design-based variance estimation with modern DiD influence functions, citing Binder (1983), Rao & Wu (1988), Shao (1996)
- **Research-grade survey DGP** — 8 new parameters on `generate_survey_did_data()`: `icc`, `weight_cv`, `informative_sampling`, `heterogeneous_te_by_strata`, `te_covariate_interaction`, `covariate_effects`, `strata_sizes`, `return_true_population_att`. All backward-compatible.
- **R validation expansion** — 4 additional estimators cross-validated against R's `survey::svyglm()`: ImputationDiD, StackedDiD, SunAbraham, TripleDifference. Survey R validation coverage now 8 of 16 estimators.
- **LinkedIn carousel** for Wooldridge ETWFE estimator announcement

### Changed
- Survey tutorial rewritten: leads with "Why Survey Design Matters" section showing flat-weight vs design-based comparison with known ground truth, coverage simulation, and false pre-trend detection rates
- Documentation refresh: ROADMAP.md, llms.txt, llms-full.txt, llms-practitioner.txt, choosing_estimator.rst updated for v2.9.0 — added WooldridgeDiD and StaggeredTripleDifference, DDD flowchart branch, standardized estimator counts, qualified survey claims
- Survey roadmap updated: Phase 10a-10d marked shipped, conditional PT noted for 10e

### Fixed
- Fix stale "EfficientDiD covariates + survey not supported" note in choosing_estimator.rst
- Fix WooldridgeDiD described as "ASF-based" for OLS path (OLS uses direct coefficients; ASF only for logit/Poisson)
- Fix dead StaggeredTripleDifference API link in llms.txt
- Fix survey example attribute: `.design_effect` not `.deff` in llms-full.txt
- Fix `subpopulation()` example to show tuple unpacking in llms-full.txt
- Remove 8 resolved items from TODO.md

## [2.9.0] - 2026-04-04

### Added
- **WooldridgeDiD (ETWFE)** estimator — Extended Two-Way Fixed Effects from Wooldridge (2025, 2023). Supports OLS, logit, and Poisson QMLE paths with ASF-based ATT and delta-method SEs. Four aggregation types (simple, group, calendar, event) matching Stata `jwdid_estat`. Alias: `ETWFE`. (PR #216, thanks @wenddymacro)
- **EfficientDiD survey + covariates** — doubly robust covariate path now threads survey weights through all four nuisance estimation stages (outcome regression, propensity ratio sieve, inverse propensity sieve, kernel-smoothed conditional Omega*). Previously raised `NotImplementedError`.
- **Survey real-data validation** (Phase 9) — 15 cross-validation tests against R's `survey` package using three real federal survey datasets:
  - **API** (R `survey` package): TSL variance with strata, FPC, subpopulations, covariates, and Fay's BRR replicates
  - **NHANES** (CDC/NCHS): TSL variance with strata + PSU + nest=TRUE, validating the ACA young adult coverage provision DiD
  - **RECS 2020** (U.S. EIA): JK1 replicate weight variance with 60 pre-computed replicate columns
  - ATT, SE, df, and CI match R to machine precision (< 1e-10) where directly comparable; known deviations documented in REGISTRY.md (TWFE SE differs due to unit FE absorption; subpopulation df differs due to strata preservation)
- **Label-gated CI** — test workflows now require `ready-for-ci` label before running, reducing wasted CI during AI review rounds. AI review workflow always runs.
- **Documentation dependency map** (`docs/doc-deps.yaml`) — maps source files to impacted documentation. New `/docs-impact` skill flags which docs need updating when source files change.

### Changed
- WooldridgeDiD: full interacted covariate basis (D_g × X, f_t × X) for OLS path
- `/submit-pr`, `/push-pr-update`, `/pre-merge-check`, `/docs-check` skills updated for label-gated CI and doc-deps workflow

### Fixed
- Fix WooldridgeDiD OLS unbalanced demeaning and nonlinear never-treated identification
- Fix WooldridgeDiD Poisson dropped-cell bug and anticipation propagation
- Fix EfficientDiD IF-scale mismatch in survey aggregation and zero-weight never-treated guard
- Fix bootstrap clustering and delta-method reduced space in WooldridgeDiD

## [2.8.4] - 2026-04-04

### Added
- **SDR replicate method** (Phase 8a) — Successive Difference Replication for ACS PUMS users. `SurveyDesign(replicate_method="SDR")` with variance formula `V = 4/R * sum((theta_r - theta)^2)`.
- **FPC support for ImputationDiD and TwoStageDiD** (Phase 8b) — finite population correction now threaded through TSL variance for both estimators.
- **Lonely PSU "adjust" in bootstrap** (Phase 8d) — `lonely_psu="adjust"` now works with survey-aware bootstrap (previously raised `NotImplementedError`). Uses Rust & Rao (1996) grand-mean centering.
- **CV on estimates** (Phase 8e) — `coef_var` property on all results objects (SE/estimate). Handles edge cases (SE=0, estimate=0).
- **Weight trimming utility** (Phase 8e) — `trim_weights(data, weight_col, upper=None, lower=None, quantile=None)` in `prep.py` for capping extreme survey weights.
- **ImputationDiD pretrends + survey** (Phase 8e) — pre-trends F-test now survey-aware using subpopulation approach for correct variance under complex designs.
- Updated ImputationDiD tutorial to demonstrate `pretrends=True` event study
- Updated survey tutorial: narrative improvements, chart rendering fixes

### Fixed
- Fix survey pretrend F-test df calculation and rank-deficient survey VCV handling
- Fix `trim_weights` NaN poisoning when weight column contains missing values
- Fix single-singleton PSU warning for lonely_psu="adjust"

## [2.8.3] - 2026-04-02

### Added
- **Silent operation warnings** — 8 operations that previously altered analysis results without informing the user now emit `UserWarning`:
  - TROP lstsq → pseudo-inverse numerical fallback
  - TwoStageDiD NaN masking of unidentified fixed effects (zeroed out with treatment indicator)
  - TwoStageDiD always-treated unit removal (sample size change)
  - CallawaySantAnna silent (g,t) pair skipping (zero treated or control observations)
  - TROP missing treatment indicator fill with 0 (control)
  - Rust → Python backend fallback (previously debug log only)
  - Survey weight normalization (pweights/aweights rescaled to mean=1)
  - `np.inf` → 0 never-treated convention conversion
- **ImputationDiD pre-period event study coefficients** — pre-treatment "effects" (should be ~0 under parallel trends) for visual pre-trends assessment, following BJS (2024) Test 1
- **TwoStageDiD pre-period event study coefficients** — same pre-trends extension
- **Replicate weight expansion** to 7 additional estimators: DifferenceInDifferences, TwoWayFixedEffects, MultiPeriodDiD, SunAbraham, StackedDiD, ImputationDiD, TwoStageDiD (coverage: 4/13 → 11/13)

### Changed
- ImputationDiD pre-period coefficients use BJS Test 1 (impute Y(0) for treated units in pre-treatment periods)
- SunAbraham replicate weights use full interaction-weighted refit per replicate with cohort-level SEs

### Fixed
- Fix zero-weight demeaning safety in replicate weight paths
- Fix `df_survey` writeback for rank-deficient replicate designs (df=0)
- Fix ImputationDiD `balance_e` zero-qualifying-cohort fallback in pretrends path
- Fix survey zero-mass (g,t) skip warning gap
- Fix SunAbraham positional assignment in replicate loop

## [2.8.2] - 2026-04-02

### Added
- **EPV diagnostics for propensity score logit** — events-per-variable (EPV) checks with Peduzzi convention (predictors excluding intercept) for CallawaySantAnna IPW/DR, TripleDifference IPW/DR, and StaggeredTripleDifference
- `epv_summary()` / `epv_diagnostics` on post-fit results for CallawaySantAnna, TripleDifference, and StaggeredTripleDifference
- `diagnose_propensity()` pre-estimation helper on CallawaySantAnna
- EPV summary block in TripleDifference `summary()` output
- `epv_threshold` parameter for propensity score estimation — warns on low EPV (default) or escalates via `rank_deficient_action="error"`

### Changed
- Default propensity score fallback behavior: safer defaults with method-specific warning messages
- EPV denominator uses predictor count excluding intercept (Peduzzi et al. 1996 convention)

### Fixed
- Fix TripleDifference survey-weighted fallback propensity score
- Fix NaN cache poisoning in propensity score estimation
- Fix `epv_summary` column schema on empty results
- Fix SDDD EPV: use min-EPV across comparison cohorts with cache diagnostic propagation
- Fix `diagnose_propensity` `np.inf` handling

## [2.8.1] - 2026-04-01

### Added
- **Survey-aware DiD tutorial** (`docs/tutorials/16_survey_did.ipynb`) — Phase 7c complete. Full workflow with strata, PSU, FPC, replicate weights, subpopulation analysis, and DEFF diagnostics. Includes `generate_survey_did_data()` DGP function.
- **Survey R cross-validation** — benchmark scripts and tests comparing TSL variance against R's `survey::svyglm()` for basic DiD and TWFE with full survey designs (strata, PSU, FPC). Committed JSON fixtures for CI without R.
- **HonestDiD methodology review and validation** — 478 lines of methodology tests, paper review document, rewritten optimal FLCI with first-difference reparameterization.
- **StaggeredTripleDifference survey support** — full `SurveyDesign` integration with strata/PSU/FPC, replicate weights, and survey-aware bootstrap.

### Changed
- HonestDiD: rewrite optimal FLCI with proper first-difference reparameterization and centrosymmetric LP optimization
- HonestDiD: use `conf_int` from results instead of hardcoded `1.96*se` in event study plots
- Survey tutorial cross-referenced from choosing_estimator.rst and quickstart.rst

### Fixed
- Fix HonestDiD identified set computation and inference (F1-F6 from Rambachan & Roth 2023)
- Fix FLCI slope count (T not T-1) and constraint formula
- Fix NaN CI misclassification as significant (P0 finding)
- Fix M=0 linear extrapolation and survey df folded nct in REGISTRY.md
- Fix replicate-weight scale invariance and BRR test fixtures
- Fix JK1 populated-PSU guard and narrow warning filter

## [2.8.0] - 2026-03-31

### Added
- **Staggered Triple Difference estimator** (Ortiz-Villavicencio & Sant'Anna 2025)
  - `StaggeredTripleDifference` class with group-time ATT(g,t) for DDD designs with staggered adoption
  - Event study aggregation, pre-treatment placebo effects, multiplier bootstrap inference
  - R benchmark validation against `triplediff` package
  - DGP function `generate_staggered_ddd_data()` for simulation and testing
- **Survey Phase 7a: CS IPW/DR + covariates + survey**
  - DRDID panel nuisance-estimation IF corrections (PS + OR) under survey weights
  - Survey-weighted propensity score estimation and outcome regression
  - IFs account for nuisance parameter estimation uncertainty (Sant'Anna & Zhao 2020, Theorem 3.1)
- **Survey Phase 7b: Repeated cross-sections**
  - `CallawaySantAnna(panel=False)` for repeated cross-section surveys (BRFSS, ACS, CPS)
  - Cross-sectional DRDID: `reg` matches `DRDID::reg_did_rc`, `dr` matches `DRDID::drdid_rc`, `ipw` matches `DRDID::std_ipw_did_rc`
  - Survey weights, covariates, and all estimation methods supported
- **Survey Phase 7d: HonestDiD + survey variance**
  - Survey df and full event-study VCV from IF vectors propagated to sensitivity analysis
  - t-distribution critical values with survey degrees of freedom
  - Bootstrap/replicate designs fall back to diagonal VCV with warning
- **Plotly visualization styling**: thread `marker`, `markersize`, `linewidth`, `capsize`, `ci_linewidth` kwargs through plotly backends (previously silently ignored)
- AI agent discoverability for practitioner guide

### Changed
- HonestDiD now raises `ValueError` on non-consecutive event-time grid (was warning)
- HonestDiD validates full grid around reference period
- Panel IPW/DR PS correction scaling matches R's `H/n`, `asy_rep/n`, `colMeans` convention
- RC IF normalization follows R's `psi` convention with explicit `phi` conversion

### Fixed
- Fix HonestDiD reference-aware pre/post split for varying-base event studies
- Fix HonestDiD `_estimate_max_pre_violation` to use reference-aware pre_periods
- Fix panel M2 gradient scaling for IPW/DR nuisance IF corrections
- Fix VCV index alignment for repeated cross-section aggregation
- Fix replicate-weight df propagation: return per-statistic df instead of mutating shared state
- Fix WIF population consistency: zero df `first_treat` for ineligible units
- Fix bootstrap RCS cohort-mass weighting and stale event-study VCV reset

## [2.7.6] - 2026-03-28

### Added
- **AI practitioner guardrails** based on Baker et al. (2025) "Difference-in-Differences Designs: A Practitioner's Guide"
  - `practitioner.py` module with 8-step workflow enforcement for AI agents
  - Estimator-specific handlers ensuring correct diagnostic ordering (pre-trends before estimation, Bacon decomposition before estimator selection)
  - `docs/llms.txt`, `docs/llms-practitioner.txt`, `docs/llms-full.txt` for AI agent discoverability
  - Evaluation rubric (`docs/practitioner-guide-evaluation.md`) with correctness-aware scoring
- **Survey Phase 6: Advanced features**
  - Survey-aware bootstrap for all 8 bootstrap-using estimators (PSU-level multiplier for CS/Imputation/TwoStage/Continuous/Efficient; Rao-Wu rescaled for SA/SyntheticDiD/TROP)
  - Replicate weight variance estimation (BRR, Fay's BRR, JK1, JKn) for OLS-based and IF-based estimators
  - Per-coefficient DEFF diagnostics comparing survey vs SRS variance
  - Subpopulation analysis via `SurveyDesign.subpopulation()` preserving full design structure
  - CS analytical expansion: strata/PSU/FPC for aggregated SEs via `compute_survey_if_variance()`
  - TROP cross-classified pseudo-strata for survey-aware bootstrap

### Changed
- Estimator-specific guidance for parallel trends tests and placebo checks (no shared templates)
- SDiD and TROP split into separate decision tree branches in practitioner workflow

### Fixed
- Fix replicate weight df calculation using pivoted QR rank with R-compatible tolerance
- Fix replicate IF variance score scaling for EfficientDiD, TripleDiff, ContinuousDiD
- Fix panel-to-unit replicate weight propagation and normalization
- Fix CS zero-mass return type and vectorized guard for survey paths
- Fix `solve_logit` effective-sample validation for zero-weight designs
- Fix subpopulation mask validation and EfficientDiD bootstrap guard

## [2.7.5] - 2026-03-23

### Added
- **Phase 4 survey support** for ImputationDiD, TwoStageDiD, and CallawaySantAnna estimators
  - ImputationDiD/TwoStageDiD: analytical survey inference with weights, strata, and PSU (FPC not supported; bootstrap+survey deferred)
  - CallawaySantAnna: weights-only analytical IF/WIF inference matching R `did::wif()` (strata/PSU/FPC deferred)
  - Survey-aware aggregation for group-time, event-study, and overall ATT
- **EfficientDiD enhancements**: doubly robust covariates path, sieve inverse propensity (Eq 3.12), conditional Omega*
- **Cluster-robust SEs** for EfficientDiD with last-cohort control and Hausman pretest
- **Enhanced visualizations**: synth weights, staircase, dose-response, group-time heatmap, plotly backend
- **Local AI review skill** (`/ai-review-local`) with Responses API, delta-diff re-review, and cost visibility
- Add `plotly` optional dependency group (`pip install diff-diff[plotly]`)

### Changed
- Migrate AI local review from Chat Completions to Responses API
- Split TROP estimator into mixin modules (`trop_local.py`, `trop_global.py`) for maintainability
- Refactor `visualization.py` into `visualization/` subpackage
- Improve review script: full-file context, content-first parsing, tiered matching, fingerprint stability

### Fixed
- Fix CallawaySantAnna reg+cov control IF normalization and survey df calculation
- Fix TripleDifference TSL double-weighting and RA nuisance linearization with survey weights
- Fix ContinuousDiD bread normalization, fweight TSL scaling, and weighted-mass IF linearization
- Fix BaconDecomposition exact-weight survey unit_share and empty-cell guard
- Fix SunAbraham survey weight floor in overall ATT aggregation
- Fix plotly event study for non-numeric periods, heatmap masking, color parser

## [2.7.4] - 2026-03-21

### Added
- **Survey/sampling weights support** (`survey_design` parameter) for `DifferenceInDifferences` and `TwoWayFixedEffects`
  - Taylor-series linearization (TSL) variance estimation with stratified multi-stage designs
  - Probability weights (pweight), frequency weights (fweight), and analytic weights (aweight)
  - Finite population correction (FPC) support
  - PSU-based clustering with lonely PSU handling
  - New `diff_diff/survey.py` module with `SurveyDesign` and `compute_survey_vcov`
- **EfficientDiD validation tests** against Chen, Sant'Anna & Xie (2025) using HRS dataset
  - HRS validation fixture with provenance documentation
  - Shared DGP helper in `tests/helpers/edid_dgp.py`
- Simulation-based power analysis for all registry-backed estimators (MDE, sample size, power curves); unregistered estimators supported via custom `data_generator` and `result_extractor`

### Changed
- Extend power analysis to support all registry-backed estimators with `result_extractor` parameter
- Update power analysis tutorial with simulation-based features
- Reject `absorb + fixed_effects` combination (FWL violation) in both survey and non-survey paths

### Fixed
- TWFE cluster-as-PSU injection for no-PSU survey designs
- Non-unique PSU labels across strata with `nest=False`
- FPC validation moved to `compute_survey_vcov` for effective PSU structure
- Survey HC1 meat formula and weighted rank-deficiency handling
- Zero-SE inference, full-census FPC, fweight contract corrections
- Bootstrap+survey fallback in MultiPeriodDiD
- DDD `_snap_n` floor mismatch and `n_per_cell` suppression scope

## [2.7.3] - 2026-03-19

### Added
- Add aarch64 Linux wheel builds to publish workflow

### Changed
- Improve documentation information architecture
- Fix silent interpreter skip and consolidate Linux jobs in publish workflow

## [2.7.2] - 2026-03-18

### Added
- SEO infrastructure: meta tags, sitemap, llms.txt/llms-full.txt for AI discoverability

### Changed
- Rename TROP `method="twostep"` to `method="local"`; `"twostep"` deprecated, removal in v3.0
- Rename internal TROP `_joint_*` methods to `_global_*` for consistency

### Fixed
- Fix TROPResults schema: report unit counts not observation counts
- Fix llms-full.txt accuracy and dynamic canonical URLs

## [2.7.1] - 2026-03-15

### Changed
- Replace BFGS logit with IRLS for propensity score estimation in CallawaySantAnna
- Reject `pscore_trim=0.0` to prevent infinite IPW weights
- Honor `rank_deficient_action="error"` in propensity score paths
- Validate `pscore_trim` at `fit()` to guard against `set_params` bypass
- Mark slow tests (`@pytest.mark.slow`) and exclude by default for faster local iteration
- Use per-class slow markers in `test_trop.py` for faster pure Python CI

### Fixed
- Vectorize Sun-Abraham bootstrap resampling loop for improved performance

## [2.7.0] - 2026-03-15

### Added
- **EfficientDiD estimator** (`EfficientDiD`) implementing Chen, Sant'Anna & Xie (2025) efficient DiD
- CallawaySantAnna event study SEs (WIF-based) and simultaneous confidence bands (sup-t)
- R comparison tests for event-study SEs and cband critical values
- Non-finite outcome validation in `EfficientDiD.fit()`
- CallawaySantAnna speed benchmarks with baseline results
- Estimator alias documentation in README, quickstart, and API docs

### Changed
- **BREAKING: TROP nuclear norm solver step size fix** — The proximal gradient
  threshold for the L matrix (both `method="global"` and `method="twostep"` with
  finite `lambda_nn`) was over-shrinking singular values by a factor of 2. The
  soft-thresholding threshold was λ_nn/max(δ) when the correct value is
  λ_nn/(2·max(δ)), derived from the Lipschitz constant L_f=2·max(δ) of the
  quadratic gradient. This fix produces higher-rank L matrices and closer
  agreement with exact convex optimization solutions. Users with finite
  `lambda_nn` will observe different ATT estimates. Added FISTA/Nesterov
  acceleration to the twostep inner solver for faster L convergence.
- Add (1-W) weight masking to TROP global method, rename joint→global
- Optimize CallawaySantAnna covariate path with Cholesky and pscore caching
- Update Codex AI review model from gpt-5.2-codex to gpt-5.4

### Fixed
- Fix CallawaySantAnna event study SEs (missing WIF) and simultaneous confidence bands
- Fix analytical and bootstrap WIF pg scaling to use global N
- Fix TROP nuclear norm solver threshold scaling for non-uniform weights
- Fix stale coefficients in TROP global low-rank solver and NaN bootstrap poisoning
- Fix NaN-cell preservation in CallawaySantAnna balance_e aggregation
- Fix not-yet-treated cache keys and dropped-cell warning
- Fix rank-deficiency handling with Cholesky rank checks and reduced-column solve
- Fix Rust convergence criterion, n_valid_treated consistency, and NaN bootstrap SE

## [2.6.1] - 2026-03-08

### Added
- Short aliases for all estimators (e.g., `DiD`, `TWFE`, `EventStudy`, `CS`, `SDiD`)

### Changed
- Update roadmap for v2.6.0: reflect completed work and refresh priorities
- Add ContinuousDiD to ReadTheDocs API reference and choosing guide
- Add SPT identification caveat and data requirements per review
- Add time-invariant dose requirement to data requirements

### Fixed
- Fix alias docs wording: clarify TROP has no alias
- Fix ContinuousDiD SE method: influence function, not delta method
- Fix methodology doc: influence functions, not delta method for ContinuousDiD SEs
- Fix dollar sign escaping in continuous DiD tutorial
- Fix continuous DiD tutorial formatting: escape dollar signs and split chart cell
- Fix methodology claims and slide numbering per PR review

## [2.6.0] - 2026-02-22

### Added
- **Continuous DiD estimator** (`ContinuousDiD`) implementing Callaway, Goodman-Bacon & Sant'Anna (2024)
  for continuous treatment dose-response analysis
  - `ContinuousDiDResults` with dose-response curves and event-study effects
  - `DoseResponseCurve` with bootstrap p-values
  - Analytical and bootstrap event-study SEs
  - P(D=0) warning for low-probability control groups
- Stacked DiD tutorial (Tutorial 13) with Q-weight computation walkthrough

### Changed
- Clarify aggregate Q-weight computation for unbalanced panels in Stacked DiD tutorial
- Replace SunAbraham manual bootstrap stats with NaN-gated utility

### Fixed
- Fix not-yet-treated control mask to respect anticipation parameter in ContinuousDiD
- Guard non-finite `original_effect` in `compute_effect_bootstrap_stats`
- Fix bootstrap NaN propagation for rank-deficient cells
- Fix NaN propagation in rank-deficient spline predictions
- Guard bootstrap NaN propagation: SE/CI/p-value all NaN when SE invalid
- Fix bootstrap ACRT^{glob} centering bug
- Fix bootstrap percentile inference and analytical event-study SE scaling
- Fix control group bug and dose validation in ContinuousDiD

## [2.5.0] - 2026-02-19

### Added
- Stacked DiD estimator (`StackedDiD`) implementing Wing, Freedman & Hollingsworth (2024)
  with corrective Q-weights for compositional balance across event times
- Sub-experiment construction per adoption cohort with clean (never-yet-treated) controls
- IC1/IC2 trimming for compositional balance across event times
- Q-weights for aggregate, population, or sample share estimands (Table 1)
- WLS event study regression via sqrt(w) transformation
- `stacked_did()` convenience function
- R benchmark scripts for Stacked DiD validation (`benchmarks/R/benchmark_stacked_did.R`)
- Comprehensive test suite for Stacked DiD (`tests/test_stacked_did.py`)

### Fixed
- NaN inference handling in pure Python mode for edge cases

## [2.4.3] - 2026-02-19

### Changed
- Rewrite TripleDifference estimator to match R's `triplediff::ddd()` — all 3 estimation
  methods (DR, IPW, RA) now use three-DiD decomposition with influence function SE, achieving
  <0.001% relative difference from R across all 24 comparisons (4 DGPs × 3 methods × 2 covariate settings)
- Validate cluster column in TripleDifference for proper cluster-robust SEs
- Handle non-finite influence function propagation in TripleDifference edge cases
- Propensity score fallback uses Hessian-based SE when score optimization fails
- Improved R-squared consistency across estimation methods

### Fixed
- Fix low cell count warning and overlap detection in TripleDifference IPW
- Fix cluster SE computation to use functional (groupby) approach instead of loop
- Fix rank deficiency handling in TripleDifference regression adjustment

### Added
- 91 methodology verification tests for TripleDifference (`tests/test_methodology_triple_diff.py`)
- R benchmark scripts for triple difference validation (`benchmarks/R/benchmark_triplediff.R`)
- Update METHODOLOGY_REVIEW.md to reflect completed TripleDifference review

## [2.4.2] - 2026-02-18

### Added
- **Conditional BLAS linking for Rust backend** — Apple Accelerate on macOS, OpenBLAS on Linux.
  Pre-built wheels now use platform-optimized BLAS for matrix-vector and matrix-matrix
  operations across all Rust-accelerated code paths (weights, OLS, TROP). Windows continues
  using pure Rust (no external dependencies). Improves Rust backend performance at larger scales.
- `rust_backend_info()` diagnostic function in `diff_diff._backend` — reports compile-time
  BLAS feature status (blas, accelerate, openblas)

### Fixed
- **Rust SDID backend performance regression at scale** — Frank-Wolfe solver was 3-10x slower than pure Python at 1k+ scale
  - Gram-accelerated FW loop for time weights: precomputes A^T@A, reducing per-iteration cost from O(N×T0) to O(T0) (~100x speedup per iteration at 5k scale)
  - Allocation-free FW loop for unit weights: 1 GEMV per iteration (was 3), zero heap allocations (was ~8)
  - Dispatch based on problem dimensions: Gram path when T0 < N, standard path when T0 >= N
  - Rust backend now faster than pure Python at all scales

## [2.4.1] - 2026-02-17

### Added
- Tutorial notebook for Two-Stage DiD (Gardner 2022) (`docs/tutorials/12_two_stage_did.ipynb`)

### Changed
- Module splits for large files: ImputationDiD, TwoStageDiD, and TROP each split into separate results and bootstrap submodules
- Migrated remaining inline inference computations to `safe_inference()` utility
- Replaced `@` operator with `np.dot()` at observation-dimension sites to avoid Apple M4 BLAS warnings
- Updated TODO.md and ROADMAP.md for accuracy post-v2.4.0

### Fixed
- Matplotlib import guards added to tutorials 11 and 12
- Various bug fixes from code quality cleanup (diagnostics, estimators, linalg, staggered, sun_abraham, synthetic_did, triple_diff)

## [2.4.0] - 2026-02-16

### Added
- **Gardner (2022) Two-Stage DiD estimator** (`TwoStageDiD`)
  - Two-stage estimator: (1) estimate unit+time FE on untreated obs, (2) regress residualized outcomes on treatment indicators
  - `TwoStageDiDResults` with overall ATT, event study, group effects, per-observation treatment effects
  - `TwoStageBootstrapResults` for multiplier bootstrap inference on GMM influence function
  - `two_stage_did()` convenience function for quick estimation
  - Point estimates identical to ImputationDiD; different variance estimator (GMM sandwich vs. conservative)
  - No finite-sample adjustments (raw asymptotic sandwich, matching R `did2s`)
- Proposition 5 detection for unidentified long-run horizons without never-treated units

### Changed
- Workflow improvements to reduce PR review rounds

### Fixed
- Zero-observation horizons/cohorts producing se=0 instead of NaN in TwoStageDiD
- Edge case fixes for TwoStageDiD (PR review feedback)
- Grep PCRE patterns updated to use POSIX character classes

## [2.3.2] - 2026-02-16

### Added
- **Python 3.13 support** with upper version cap (`>=3.9,<3.14`)

### Changed
- **Sun-Abraham methodology review** (PR #153)
  - IW aggregation weights now use event-time observation counts (not group sizes)
  - Normalize `np.inf` never-treated encoding before treatment group detection
  - Add R benchmark scripts and methodology-aligned tests
- Use `rank_deficient_action` and `np.errstate` instead of broad `RuntimeWarning` filter in SDID tutorial

### Fixed
- Sun-Abraham bootstrap NaN propagation for non-finite ATT estimates
- Sun-Abraham df_adjustment off-by-one in analytical SE computation
- CI pandas compatibility for SunAbraham bootstrap inference
- SyntheticDiD tutorial: eliminate pre-treatment fit warnings

## [2.3.1] - 2026-02-15

### Fixed
- Fix docs/PyPI version mismatch (issue #146) — RTD now builds versioned docs from source
- Fix RTD docs build failure caused by Rust/maturin compilation timeout on ReadTheDocs

### Changed
- Remove Rust outer-loop variance estimation for SyntheticDiD (placebo and bootstrap)
  - Fixes SE mismatch between pure Python and Rust backends (different RNG sequences)
  - Fixes Rust performance regression at 1k+ scale (memory bandwidth saturation from rayon parallelism)
  - Inner Frank-Wolfe weight computation still uses Rust when available

### Documentation
- Re-run SyntheticDiD benchmarks against R after Frank-Wolfe methodology rewrite
- Updated `docs/benchmarks.rst` SDID validation results, performance tables, and known differences
- ATT now matches R to < 1e-10 (previously 0.3% diff) since both use Frank-Wolfe optimizer

## [2.3.0] - 2026-02-09

### Added
- **Borusyak-Jaravel-Spiess (2024) Imputation DiD estimator** (`ImputationDiD`)
  - Efficient imputation estimator for staggered DiD designs
  - OLS on untreated observations for unit+time FE, impute counterfactual Y(0), aggregate
  - Conservative variance (Theorem 3) with `aux_partition` parameter for SE tightness
  - Pre-trend test (Equation 9) via `results.pretrend_test()`
  - Percentile bootstrap inference
  - Influence-function bootstrap with sparse variance and weight/covariate fixes
  - Absorbing-treatment validation for non-constant `first_treat`
  - Empty event-study warning for unidentified long-run horizons
- **`/paper-review` skill** for academic paper methodology extraction
- **`/read-feedback-revise` skill** for addressing PR review comments
- **`--pr` flag for `/review-plan` skill** to review plans posted as PR comments
- **`--updated` flag for `/review-plan` skill** for re-reviewing revised plans
- **MultiPeriodDiD vs R (fixest) benchmark** for cross-language validation

### Changed
- Shortened test suite runtime with parallel execution and reduced iterations

### Fixed
- **TWFE within-transformation bug** identified during methodology review
- TWFE: added non-{0,1} binary time warning, ATT invariance tests, and R fixture caching
- TWFE: single-pass demeaning, HC1 test fix, fixest coeftable comparison
- MultiPeriodDiD: added unit FE and NaN guard for R comparison benchmark
- Removed tracked PDF from repo and gitignored papers directory

## [2.2.1] - 2026-02-07

### Changed
- **MultiPeriodDiD: Full event-study specification** (BREAKING)
  - Treatment × period interactions now created for ALL periods (pre and post),
    not just post-treatment
  - Pre-period coefficients available for parallel trends assessment
  - Default reference period changed from first to last pre-period (e=-1 convention)
    with FutureWarning for one release cycle
  - `period_effects` dict now contains both pre and post period effects
  - `to_dataframe()` includes `is_post` column
  - `summary()` output now shows pre-period effects section
  - t_stat uses `np.isfinite(se) and se > 0` guard (consistent with other estimators)

### Added
- Time-varying treatment warning when `unit` is provided and treatment varies
  within units (guides users toward ever-treated indicator D_i)
- `unit` parameter to `MultiPeriodDiD.fit()` for staggered adoption detection
- `reference_period` and `interaction_indices` attributes on `MultiPeriodDiDResults`
- `pre_period_effects` and `post_period_effects` convenience properties on results
- Pre-period section in `summary()` output with reference period indicator
- `ValueError` when `reference_period` is set to a post-treatment period
- Staggered adoption warning when treatment timing varies across units (with `unit` param)
- Informative KeyError when accessing reference period via `get_effect()`

### Removed
- **TROP `variance_method` parameter** — Jackknife variance estimation removed.
  Bootstrap (the only method specified in Athey et al. 2025) is now always used.
  The `variance_method` field has also been removed from `TROPResults`.
- **TROP `max_loocv_samples` parameter** — Control observation subsampling removed
  from LOOCV tuning parameter selection. Equation 5 of Athey et al. (2025) explicitly
  sums over ALL control observations where D=0; the previous subsampling (default 100)
  was not specified in the paper. LOOCV now uses all control observations, making
  tuning fully deterministic. Inner LOOCV loops in the Rust backend are parallelized
  to compensate for the increased observation count.

### Fixed
- HonestDiD: filter non-finite period effects from MultiPeriodDiD results
  (prevents NaN propagation into sensitivity bounds; raises ValueError
  when no finite pre- or post-period effects remain)
- HonestDiD VCV extraction: now uses interaction sub-VCV instead of full regression VCV
  (via `interaction_indices` period → column index mapping)
- MultiPeriodDiD: `avg_se` guard now checks `np.isfinite()` (matches per-period pattern;
  prevents `avg_t_stat=0` / `avg_p_value=1` when variance is infinite)
- HonestDiD: extraction now uses explicit pre-then-post ordering instead of sorted period
  labels (prevents misclassification when period labels don't sort chronologically)
- Backend-aware test parameter scaling for pure Python CI performance
- Lower TROP stratified bootstrap threshold floor from 11 to 5 for pure Python CI

## [2.2.0] - 2026-01-27

### Added
- **Windows wheel builds** using pure-Rust `faer` library for linear algebra (PR #115)
  - Eliminates external BLAS/LAPACK dependencies (no OpenBLAS or Intel MKL required)
  - Enables cross-platform wheel builds for Linux, macOS, and Windows
  - Simplifies installation on all platforms

### Changed
- **Rust backend migrated from nalgebra/ndarray to faer** (PR #115)
  - OLS solver now uses faer's SVD implementation
  - Robust variance estimation uses faer's matrix operations
  - TROP distance calculations use faer primitives
  - Maintains numerical parity with existing NumPy backend

### Fixed
- **Rust backend numerical stability improvements** (PR #115)
  - Improved singular matrix detection with condition number checks
  - NaN propagation in variance-covariance estimation
  - Fallback to Python backend on numerical instability with warning
  - Underdetermined SVD handling (n < k case)
- **macOS CI compatibility** for Python 3.14 with `PYO3_USE_ABI3_FORWARD_COMPATIBILITY`

## [2.1.9] - 2026-01-26

### Added
- **Unified LOOCV for TROP joint method** with Rust acceleration (PR #113)
  - Leave-one-out cross-validation for rank and regularization parameter selection
  - Rust backend provides significant speedup for LOOCV grid search

### Fixed
- **TROP joint method Rust/Python parity** (PR #113)
  - Fixed valid_count bug in LOOCV computation
  - Proper NaN exclusion for units with no valid pre-period data
  - Zero weight assignment for units missing pre-period data
  - Jackknife variance estimation fixes
  - Staggered adoption validation and simultaneous adoption enforcement
  - Treated-pre NaN handling improvements
  - LOOCV subsampling fix for Python-only path

## [2.1.8] - 2026-01-25

### Added
- **`/push-pr-update` skill** for committing and pushing PR revisions
  - Commits local changes to current branch and pushes to remote
  - Triggers AI code review automatically
  - Robust handling for fork repos, unpushed commits, and upstream tracking

### Fixed
- **TROP estimator methodology alignment** (PR #110)
  - Aligned with paper methodology (Equation 5, D matrix semantics)
  - NaN propagation and LOOCV warnings improvements
  - Rust backend test alignment with new loocv_grid_search return signature
  - LOOCV cycling, D matrix validation fixes
  - Final estimation infinity handling and edge case fixes
  - Absorbing-state gap detection and n_post_periods fix

### Changed
- **`/submit-pr` skill improvements** (PR #111)
  - Case-insensitive secret scanning with POSIX ERE regex
  - Verify origin ref exists before push
  - Dynamic default branch detection with fallback
  - Robust handling for unpushed commits, fork repos
  - Files count display in PR summary

## [2.1.7] - 2026-01-25

### Fixed
- **`plot_event_study` reference period normalization behavior**
  - Effects are now only normalized when `reference_period` is explicitly provided
  - Auto-inferred reference periods only apply hollow marker styling (no normalization)
  - Reference period SE is set to NaN during normalization (constraint, not estimate)
  - Updated docstring to clarify explicit vs auto-inferred behavior

### Changed
- Refactored visualization tests to reuse `cs_results` fixture for better performance

## [2.1.6] - 2026-01-24

### Added
- **Methodology verification tests** for DifferenceInDifferences estimator
  - Comprehensive test suite validating all REGISTRY.md requirements
  - Tests for formula interface, coefficient extraction, rank deficiency handling
  - Singleton cluster variance estimation behavioral tests

### Changed
- **REGISTRY.md documentation improvements**
  - Clarified singleton cluster formula notation (u_i² X_i X_i' instead of ambiguous residual² × X'X)
  - Verified DifferenceInDifferences behavior against documented requirements

## [2.1.5] - 2026-01-22

### Added
- **METHODOLOGY_REVIEW.md** tracking document for methodology review progress
  - Review status summary table for all 12 estimators
  - Detailed notes template for each estimator by category
  - Review process guidelines with checklist and priority ordering
- **`base_period` parameter** for CallawaySantAnna pre-treatment effect computation
  - "varying" (default): Pre-treatment uses t-1 as base (consecutive comparisons)
  - "universal": All comparisons use g-anticipation-1 as base
  - Matches R `did::att_gt()` base_period parameter
- **Pre-merge-check skill** (`/pre-merge-check`) for automated PR validation
  - Pattern checks for NaN handling consistency
  - Context-specific checklist generation

### Changed
- **Tutorial 02 improvements**: Added pre-trends section, clarified base_period interaction with anticipation

### Fixed
- Not-yet-treated control group now properly excludes cohort g when computing ATT(g,t)
- Aggregation t_stat uses NaN (not 0.0) when SE is non-finite or zero
- Bootstrap inference for pre-treatment effects with `base_period="varying"`
- NaN propagation for empty post-treatment effects in CallawaySantAnna
- Grep word boundary pattern in pre-merge-check skill

## [2.1.4] - 2026-01-20

### Added
- **Development checklists and workflow improvements** in `CLAUDE.md`
  - Estimator inheritance map showing class hierarchy for `get_params`/`set_params`
  - Test writing guidelines for fallback paths, parameters, and warnings
  - Checklists for adding parameters and warning/error handling
- **R-style rank deficiency handling** across all estimators
  - `rank_deficient_action` parameter: "warn" (default), "error", or "silent"
  - Dropped columns have NaN coefficients (like R's `lm()`)
  - VCoV matrix has NaN for rows/cols of dropped coefficients
  - Propagated to all estimators: DifferenceInDifferences, MultiPeriodDiD, TwoWayFixedEffects, CallawaySantAnna, SunAbraham, TripleDifference, TROP, SyntheticDiD

### Fixed
- `get_params()` now includes `rank_deficient_action` parameter (fixes sklearn cloning)
- NaN vcov fallback in Rust backend for rank-deficient matrices
- MultiPeriodDiD vcov/df computation for rank-deficient designs
- Average ATT inference for rank-deficient designs

### Changed
- Rank tolerance aligned with R's `lm()` default for consistent behavior

## [2.1.3] - 2026-01-19

### Fixed
- TROP estimator paper conformance issues (Athey et al. 2025)
  - Control set now includes pre-treatment observations of eventually-treated units (Issue A)
  - Unit distance computation excludes target period per Equation 3 (Issue B)
  - Nuclear norm update uses weighted proximal gradient instead of unweighted soft-thresholding (Issue C)
  - Bootstrap sampling now stratifies by treatment status per Algorithm 3 (Issue D)
- TROP Rust backend alignment with paper specification
  - Weight normalization to sum to 1 (probability weights)
  - Weighted proximal gradient for L update with step size η ≤ 1/max(W)

### Changed
- Cleaned up unused parameters from TROP Rust API
  - Removed `control_unit_idx` and `unit_dist_matrix` from public functions
  - Per-observation distances now computed dynamically (more accurate, slightly slower)

## [2.1.2] - 2026-01-19

### Added
- **Consolidated DGP functions** in `prep.py` for all supported DiD designs
  - `generate_did_data()` - Basic 2x2 DiD data generation
  - `generate_staggered_data()` - Staggered adoption data for Callaway-Sant'Anna/Sun-Abraham
  - `generate_factor_data()` - Factor model data for TROP/SyntheticDiD
  - `generate_ddd_data()` - Triple Difference (DDD) design data
  - `generate_panel_data()` - Panel data with optional parallel trends violations
  - `generate_event_study_data()` - Event study data with simultaneous treatment

### Changed
- **Clean up development tracking files** for v2.1.1 release
  - Removed completed items from TODO.md (now tracked in CHANGELOG)
  - Updated ROADMAP.md version numbers and removed shipped TROP section
  - Updated `prep.py` line count in Large Module Files table (1338 → 1993)

## [2.1.1] - 2026-01-19

### Added
- **Rust backend acceleration for TROP estimator** delivering 5-20x overall speedup
  - `compute_unit_distance_matrix` - Parallel pairwise RMSE computation for donor matching
  - `loocv_grid_search` - Parallel leave-one-out cross-validation across 180 parameter combinations
  - `bootstrap_trop_variance` - Parallel bootstrap variance estimation
  - Automatic fallback to Python when Rust backend unavailable
  - Logging for Rust fallback events to aid debugging
- **`/bump-version` skill** for release management
  - Updates version in `__init__.py`, `pyproject.toml`, and `rust/Cargo.toml`
  - Generates CHANGELOG entries from git commits
  - Adds comparison links automatically
- **`/review-pr` skill** for code review workflow

### Changed
- **TROP estimator performance optimizations** (Python backend)
  - Vectorized distance matrix computation using NumPy broadcasting
  - Extracted tuning constants to module-level for clarity
  - Added `TROPTuningParams` TypedDict for parameter documentation

### Fixed
- Tutorial notebook validation errors in `10_trop.ipynb`
- Pre-existing RuntimeWarnings in CallawaySantAnna bootstrap (documented)
- TROP `pre_periods` parameter handling for edge cases

## [2.1.0] - 2026-01-17

### Added
- **Triply Robust Panel (TROP) estimator** implementing Athey, Imbens, Qu & Viviano (2025)
  - `TROP` class combining three robustness components:
    - Factor model adjustment via SVD (removes unobserved confounders with factor structure)
    - Synthetic control style unit weights
    - SDID style time weights
  - `TROPResults` dataclass with ATT, factors, loadings, unit/time weights
  - `trop()` convenience function for quick estimation
  - Automatic rank selection methods: cross-validation (`'cv'`), information criterion (`'ic'`), elbow detection (`'elbow'`)
  - Bootstrap and placebo-based variance estimation
  - Full integration with existing infrastructure (exports in `__init__.py`, sklearn-compatible API)
  - Tutorial notebook: `docs/tutorials/10_trop.ipynb`
  - Comprehensive test suite: `tests/test_trop.py`

**Reference**: Athey, S., Imbens, G. W., Qu, Z., & Viviano, D. (2025). "Triply Robust Panel Estimators." *Working Paper*. [arXiv:2508.21536](https://arxiv.org/abs/2508.21536)

## [2.0.3] - 2026-01-17

### Changed
- **Rust backend performance optimizations** delivering up to 32x speedup for bootstrap operations
  - Bootstrap weight generation now 16x faster on average (up to 32x for Webb distribution)
  - Direct `Array2` allocation eliminates intermediate `Vec<Vec<f64>>` (~50% memory reduction)
  - Rayon chunk size tuning (`min_len=64`) reduces parallel scheduling overhead
  - Webb distribution uses lookup table instead of 6-way if-else chain

### Added
- **LinearRegression helper class** in `linalg.py` for code deduplication
  - High-level OLS wrapper with unified coefficient extraction and inference
  - Used by DifferenceInDifferences, TwoWayFixedEffects, SunAbraham, TripleDifference
  - Provides `InferenceResult` dataclass for coefficient-level statistics
- **Cholesky factorization** for symmetric positive-definite matrix inversion in Rust backend
  - ~2x faster than LU decomposition for well-conditioned matrices
  - Automatic fallback to LU for near-singular or indefinite matrices
- **Vectorized variance computation** in Rust backend
  - HC1 meat computation: `X' @ (X * e²)` via BLAS instead of O(n×k²) loop
  - Score computation: broadcast multiplication instead of O(n×k) loop
- **Static BLAS linking options** in `rust/Cargo.toml`
  - `openblas-static` and `intel-mkl-static` features for standalone distribution
  - Eliminates runtime BLAS dependency at cost of larger binary size

## [2.0.2] - 2026-01-15

### Fixed
- **CallawaySantAnna SE computation** now exactly matches R's `did` package
  - Fixed weight influence function (wif) formula for "simple" aggregation
  - Corrected `pg` computation: uses `n_g / n_all` (matching R) instead of `n_g / total_treated`
  - Fixed wif iteration: iterates over keepers (post-treatment pairs) with individual ATT(g,t) values
  - SE difference reduced from ~2.5% to <0.01% vs R's `did` package (essentially exact match)
  - Point estimates unchanged; all existing tests pass

## [2.0.1] - 2026-01-13

### Added
- **Shared within-transformation utilities** in `utils.py`
  - `demean_by_group()` - One-way fixed effects demeaning
  - `within_transform()` - Two-way (unit + time) FE transformation
  - Reduces code duplication across `estimators.py`, `twfe.py`, `sun_abraham.py`, `bacon.py`

### Fixed
- **DataFrame fragmentation warning** - Build columns in batch instead of iteratively

### Changed
- Reverted untested Rust backend optimizations (Cholesky factorization, reduced allocations) - these will be re-added when proper testing infrastructure is available

## [2.0.0] - 2026-01-12

### Added
- **Optional Rust backend** for accelerated computation
  - 4-8x speedup for SyntheticDiD and bootstrap operations
  - Parallel bootstrap weight generation (Rademacher, Mammen, Webb)
  - Accelerated OLS solver using OpenBLAS/MKL
  - Cluster-robust variance estimation
  - Synthetic control weight optimization with simplex projection
  - Pre-built wheels for Linux x86_64 and macOS ARM64
  - Pure Python fallback for all other platforms
- **`diff_diff/_backend.py`** - Backend detection and configuration module
  - `HAS_RUST_BACKEND` flag exported in main package
  - `DIFF_DIFF_BACKEND` environment variable for backend control:
    - `'auto'` (default) - Use Rust if available, fall back to Python
    - `'python'` - Force pure Python mode
    - `'rust'` - Force Rust mode (fails if unavailable)
- **Rust source code** in `rust/` directory
  - `rust/src/lib.rs` - PyO3 module definition
  - `rust/src/bootstrap.rs` - Parallel bootstrap weight generation
  - `rust/src/linalg.rs` - OLS solver and robust variance estimation
  - `rust/src/weights.rs` - Synthetic control weights and simplex projection
- **Rust backend test suite** - `tests/test_rust_backend.py` for equivalence testing

### Changed
- Package version bumped from 1.4.0 to 2.0.0 (major version for new backend)
- CI/CD updated to build Rust extensions with maturin
- ReadTheDocs now installs from PyPI (pre-built wheels with Rust backend)

## [1.4.0] - 2026-01-11

### Added
- **Unified linear algebra backend** (`diff_diff/linalg.py`)
  - `solve_ols()` - Optimized OLS solver using scipy's gelsy LAPACK driver
  - `compute_robust_vcov()` - Vectorized (clustered) robust variance-covariance
  - Single optimization point for all estimators; prepares for future Rust backend
  - New `tests/test_linalg.py` with comprehensive tests

### Changed
- **Major performance improvements** - All estimators now significantly faster
  - BasicDiD/TWFE @ 10K: 0.835s → 0.011s (76x faster, now 4.2x faster than R)
  - CallawaySantAnna @ 10K: 2.234s → 0.109s (20x faster, now 7.2x faster than R)
  - All results numerically identical to previous versions
- **CallawaySantAnna optimizations** (`staggered.py`)
  - Pre-computed wide-format outcome matrix and cohort masks
  - Vectorized ATT(g,t) computation using numpy operations (23x faster)
  - Batch bootstrap weight generation
  - Vectorized multiplier bootstrap using matrix operations (26x faster)
- **TWFE optimization** (`twfe.py`)
  - Cached groupby indexes for within-transformation
- **All estimators migrated** to unified `linalg.py` backend
  - `estimators.py`, `twfe.py`, `staggered.py`, `triple_diff.py`,
    `synthetic_did.py`, `sun_abraham.py`, `utils.py`

### Behavioral Changes
- **Rank-deficient design matrices**: The new `gelsy` LAPACK driver handles
  rank-deficient matrices gracefully (returning a least-norm solution) rather
  than raising an explicit error. Previously, `DifferenceInDifferences` would
  raise `ValueError("Design matrix is rank-deficient")`. Users relying on this
  error for collinearity detection should validate their design matrices
  separately. Results remain numerically correct for well-specified models.

## [1.3.1] - 2026-01-10

### Added
- **SyntheticDiD placebo-based variance estimation** matching R's `synthdid` package methodology
  - New `variance_method` parameter with options `"bootstrap"` (default) and `"placebo"`
  - Placebo method implements Algorithm 4 from Arkhangelsky et al. (2021):
    1. Randomly permutes control unit indices
    2. Designates N₁ controls as pseudo-treated (matching actual treated count)
    3. Renormalizes original unit weights for remaining pseudo-controls
    4. Computes SDID estimate with renormalized weights
    5. Repeats for `n_bootstrap` replications
    6. SE = sqrt((r-1)/r) × sd(estimates)
  - Provides methodological parity with R's `synthdid::vcov(method = "placebo")`
  - `n_bootstrap` parameter now used for both bootstrap and placebo replications
  - `SyntheticDiDResults` now tracks `variance_method` and `n_bootstrap` attributes
  - Results summary displays variance method and replications count

**Reference**: Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W., & Wager, S. (2021). Synthetic Difference-in-Differences. *American Economic Review*, 111(12), 4088-4118.

## [1.3.0] - 2026-01-09

### Added
- **Triple Difference (DDD) estimator** implementing Ortiz-Villavicencio & Sant'Anna (2025)
  - `TripleDifference` class for DDD designs where treatment requires two criteria (group AND partition)
  - `TripleDifferenceResults` dataclass with ATT, SEs, cell means, and diagnostics
  - `triple_difference()` convenience function for quick estimation
  - Three estimation methods: regression adjustment (`reg`), inverse probability weighting (`ipw`), and doubly robust (`dr`)
  - Proper covariate handling (unlike naive DDD implementations that difference two DiDs)
  - Propensity score trimming for IPW/DR methods
  - Cluster-robust standard errors support
  - Tutorial notebook: `docs/tutorials/08_triple_diff.ipynb`

**Reference**: Ortiz-Villavicencio, M., & Sant'Anna, P. H. C. (2025). "Better Understanding Triple Differences Estimators." *Working Paper*. [arXiv:2505.09942](https://arxiv.org/abs/2505.09942)

## [1.2.1] - 2026-01-08

### Added
- **Expanded test coverage** for edge cases:
  - Wild bootstrap with very few clusters (< 5), including 2-3 cluster scenarios
  - Unbalanced panels with missing periods across units
  - Single treated unit scenarios for DiD and Synthetic DiD
  - Perfect collinearity detection (validates clear error messages)
  - CallawaySantAnna with single treatment cohort
  - SyntheticDiD with insufficient pre-treatment periods

### Changed
- **Refactored CallawaySantAnna bootstrap**: Extracted `_compute_effect_bootstrap_stats()` helper method for cleaner code and reduced duplication in bootstrap statistics computation.

## [1.2.0] - 2026-01-07

### Added
- **Pre-Trends Power Analysis** (Roth 2022) for assessing informativeness of pre-trends tests
  - `PreTrendsPower` class for computing power and minimum detectable violation (MDV)
  - `PreTrendsPowerResults` dataclass with power, MDV, and test statistics
  - `PreTrendsPowerCurve` for power curves across violation magnitudes
  - `compute_pretrends_power()` and `compute_mdv()` convenience functions
  - Multiple violation types: `linear`, `constant`, `last_period`, `custom`
  - Integration with Honest DiD via `sensitivity_to_honest_did()` method
  - `plot_pretrends_power()` visualization for power curves
  - Tutorial notebook: `docs/tutorials/07_pretrends_power.ipynb`
  - Full API documentation: `docs/api/pretrends.rst`

**Reference**: Roth, J. (2022). "Pretest with Caution: Event-Study Estimates after Testing for Parallel Trends." *American Economic Review: Insights*, 4(3), 305-322.

### Fixed
- **Reference period handling in pre-trends analysis**: Fixed bug where reference period was incorrectly assigned `avg_se` instead of being excluded from power calculations. Now properly excludes the omitted reference period from the joint Wald test.

## [1.1.1] - 2026-01-06

### Fixed
- **SyntheticDiD bootstrap error handling**: Bootstrap now raises clear `ValueError` when all iterations fail, instead of silently returning SE=0.0. Added warnings for edge cases (single successful iteration, high failure rate).

- **Diagnostics module error handling**: Improved error messages in `permutation_test()` and `leave_one_out_test()` with actionable guidance. Added warnings when significant iterations fail. Enhanced `run_all_placebo_tests()` to return structured error info including error type.

### Changed
- **Code deduplication**: Extracted wild bootstrap inference logic to shared `_run_wild_bootstrap_inference()` method in `DifferenceInDifferences` base class, used by both `DifferenceInDifferences` and `TwoWayFixedEffects`.

- **Type hints**: Added missing type hints to nested functions:
  - `compute_trend()` in `utils.py`
  - `neg_log_likelihood()` and `gradient()` in `staggered.py`
  - `format_label()` in `prep.py`

## [1.1.0] - 2026-01-05

### Added
- **Sun-Abraham (2021) interaction-weighted estimator** for staggered DiD
  - `SunAbraham` class implementing saturated regression approach
  - `SunAbrahamResults` with event study effects, cohort weights, and overall ATT
  - `SABootstrapResults` for bootstrap inference (SEs, CIs, p-values)
  - Support for `never_treated` and `not_yet_treated` control groups
  - Analytical and cluster-robust standard errors
  - Multiplier bootstrap with Rademacher, Mammen, or Webb weights
  - Integration with `plot_event_study()` visualization
  - Useful robustness check alongside Callaway-Sant'Anna

**Reference**: Sun, L., & Abraham, S. (2021). "Estimating Dynamic Treatment Effects in Event Studies with Heterogeneous Treatment Effects." *Journal of Econometrics*, 225(2), 175-199.

## [1.0.2] - 2026-01-04

### Changed
- Refactored `estimators.py` to reduce module size
  - Moved `TwoWayFixedEffects` to `diff_diff/twfe.py`
  - Moved `SyntheticDiD` to `diff_diff/synthetic_did.py`
  - Backward compatible re-exports maintained in `estimators.py`

### Fixed
- Fixed ReadTheDocs version display by importing from package `__version__`

## [1.0.1] - 2026-01-04

### Fixed
- Tech debt cleanup (Tier 1 + Tier 2)
  - Improved code organization and documentation
  - Fixed minor issues identified in tech debt review

## [1.0.0] - 2026-01-04

### Added
- **Goodman-Bacon decomposition** for TWFE diagnostics
  - `BaconDecomposition` class for decomposing TWFE into weighted 2x2 comparisons
  - `Comparison2x2` dataclass for individual comparisons (treated_vs_never, earlier_vs_later, later_vs_earlier)
  - `BaconDecompositionResults` with weights and estimates by comparison type
  - `bacon_decompose()` convenience function
  - `plot_bacon()` visualization for decomposition results
  - Integration via `TwoWayFixedEffects.decompose()` method
- **Power analysis** for study design
  - `PowerAnalysis` class for analytical power calculations
  - `PowerResults` and `SimulationPowerResults` dataclasses
  - `compute_mde()`, `compute_power()`, `compute_sample_size()` convenience functions
  - `simulate_power()` for Monte Carlo simulation-based power analysis
  - `plot_power_curve()` visualization for power analysis
  - Tutorial notebook: `docs/tutorials/06_power_analysis.ipynb`
- **Callaway-Sant'Anna multiplier bootstrap** for inference
  - `CSBootstrapResults` with standard errors, confidence intervals, p-values
  - Rademacher, Mammen, and Webb weight distributions
  - Bootstrap inference for all aggregation methods
- **Troubleshooting guide** in documentation
- **Standard error computation guide** explaining SE differences across estimators

### Changed
- Updated package status to Production/Stable (was Alpha)
- SyntheticDiD bootstrap now warns when >5% of iterations fail

### Fixed
- Silent bootstrap failures in SyntheticDiD now produce warnings

## [0.6.0]

### Added
- **CallawaySantAnna covariate adjustment** for conditional parallel trends
  - Outcome regression (`estimation_method='reg'`)
  - Inverse probability weighting (`estimation_method='ipw'`)
  - Doubly robust estimation (`estimation_method='dr'`)
  - Pass covariates via `covariates` parameter in `fit()`
- **Honest DiD sensitivity analysis** (Rambachan & Roth 2023)
  - `HonestDiD` class for computing bounds under parallel trends violations
  - Relative magnitudes restriction (`DeltaRM`) - bounds post-treatment violations by pre-treatment
  - Smoothness restriction (`DeltaSD`) - bounds second differences of trend violations
  - Combined restrictions (`DeltaSDRM`)
  - FLCI and C-LF confidence interval methods
  - Breakdown value computation via `breakdown_value()`
  - Sensitivity analysis over M grid via `sensitivity_analysis()`
  - `HonestDiDResults` and `SensitivityResults` dataclasses
  - `compute_honest_did()` convenience function
  - `plot_sensitivity()` for sensitivity analysis visualization
  - `plot_honest_event_study()` for event study with honest CIs
  - Tutorial notebook: `docs/tutorials/05_honest_did.ipynb`
- **API documentation site** with Sphinx
  - Full API reference auto-generated from docstrings
  - "Which estimator should I use?" decision guide
  - Comparison with R packages (did, HonestDiD)
  - Getting started / quickstart guide

### Changed
- Updated mypy configuration for better numpy type compatibility
- Modernized ruff configuration to use `[tool.ruff.lint]` section

### Fixed
- Fixed 21 ruff linting issues (import ordering, unused variables, ambiguous names)
- Fixed 94 mypy type checking issues (Optional types, numpy type casts, assertions)
- Added missing return statement in `run_placebo_test()`

## [0.5.0]

### Added
- **Wild cluster bootstrap** for valid inference with few clusters
  - Rademacher weights (default, good for most cases)
  - Webb's 6-point distribution (recommended for <10 clusters)
  - Mammen's two-point distribution
  - `WildBootstrapResults` dataclass
  - `wild_bootstrap_se()` utility function
  - Integration with `DifferenceInDifferences` and `TwoWayFixedEffects` via `inference='wild_bootstrap'`
- **Placebo tests module** (`diff_diff.diagnostics`)
  - `placebo_timing_test()` - fake treatment timing test
  - `placebo_group_test()` - fake treatment group test
  - `permutation_test()` - permutation-based inference
  - `leave_one_out_test()` - sensitivity to individual treated units
  - `run_placebo_test()` - unified dispatcher for all test types
  - `run_all_placebo_tests()` - comprehensive diagnostic suite
  - `PlaceboTestResults` dataclass
- **Tutorial notebooks** in `docs/tutorials/`
  - `01_basic_did.ipynb` - Basic 2x2 DiD, formula interface, covariates, fixed effects, wild bootstrap
  - `02_staggered_did.ipynb` - Staggered adoption with Callaway-Sant'Anna
  - `03_synthetic_did.ipynb` - Synthetic DiD with unit/time weights
  - `04_parallel_trends.ipynb` - Parallel trends testing and diagnostics
- Comprehensive test coverage (380+ tests)

## [0.4.0]

### Added
- **Callaway-Sant'Anna estimator** for staggered difference-in-differences
  - `CallawaySantAnna` class with group-time ATT(g,t) estimation
  - Support for `never_treated` and `not_yet_treated` control groups
  - Aggregation methods: `simple`, `group`, `calendar`, `event_study`
  - `CallawaySantAnnaResults` with group-time effects and aggregations
  - `GroupTimeEffect` dataclass for individual effects
- **Event study visualization** via `plot_event_study()`
  - Works with `MultiPeriodDiDResults`, `CallawaySantAnnaResults`, or DataFrames
  - Publication-ready formatting with customization options
- **Group effects visualization** via `plot_group_effects()`
- **Parallel trends testing utilities**
  - `check_parallel_trends()` - simple slope-based test
  - `check_parallel_trends_robust()` - Wasserstein distance test
  - `equivalence_test_trends()` - TOST equivalence test

## [0.3.0]

### Added
- **Synthetic Difference-in-Differences** (`SyntheticDiD`)
  - Unit weight optimization for synthetic control
  - Time weight computation for pre-treatment periods
  - Placebo-based and bootstrap inference
  - `SyntheticDiDResults` with weight accessors
- **Multi-period DiD** (`MultiPeriodDiD`)
  - Event-study style estimation with period-specific effects
  - `MultiPeriodDiDResults` with `period_effects` dictionary
  - `PeriodEffect` dataclass for individual period results
- **Data preparation utilities** (`diff_diff.prep`)
  - `generate_did_data()` - synthetic data generation
  - `make_treatment_indicator()` - create treatment from categorical/numeric
  - `make_post_indicator()` - create post-treatment indicator
  - `wide_to_long()` - reshape wide to long format
  - `balance_panel()` - ensure balanced panel data
  - `validate_did_data()` - data validation
  - `summarize_did_data()` - summary statistics by group
  - `create_event_time()` - event time for staggered designs
  - `aggregate_to_cohorts()` - aggregate to cohort means
  - `rank_control_units()` - rank controls by similarity

## [0.2.0]

### Added
- **Two-Way Fixed Effects** (`TwoWayFixedEffects`)
  - Within-transformation for unit and time fixed effects
  - Efficient handling of high-dimensional fixed effects via `absorb`
- **Fixed effects support** in base `DifferenceInDifferences`
  - `fixed_effects` parameter for dummy variable approach
  - `absorb` parameter for within-transformation approach
- **Cluster-robust standard errors**
  - `cluster` parameter for cluster-robust inference
- **Formula interface**
  - R-style formulas like `"outcome ~ treated * post"`
  - Support for covariates in formulas

## [0.1.0]

### Added
- Initial release
- **Basic Difference-in-Differences** (`DifferenceInDifferences`)
  - sklearn-like API with `fit()` method
  - Column name interface for outcome, treatment, time
  - Heteroskedasticity-robust (HC1) standard errors
  - `DiDResults` dataclass with ATT, SE, p-value, confidence intervals
  - `summary()` and `print_summary()` methods
  - `to_dict()` and `to_dataframe()` export methods
  - `is_significant` and `significance_stars` properties

[3.0.1]: https://github.com/igerber/diff-diff/compare/v3.0.0...v3.0.1
[3.0.0]: https://github.com/igerber/diff-diff/compare/v2.9.1...v3.0.0
[2.9.1]: https://github.com/igerber/diff-diff/compare/v2.9.0...v2.9.1
[2.9.0]: https://github.com/igerber/diff-diff/compare/v2.8.4...v2.9.0
[2.8.4]: https://github.com/igerber/diff-diff/compare/v2.8.3...v2.8.4
[2.8.3]: https://github.com/igerber/diff-diff/compare/v2.8.2...v2.8.3
[2.8.2]: https://github.com/igerber/diff-diff/compare/v2.8.1...v2.8.2
[2.8.1]: https://github.com/igerber/diff-diff/compare/v2.8.0...v2.8.1
[2.8.0]: https://github.com/igerber/diff-diff/compare/v2.7.6...v2.8.0
[2.7.6]: https://github.com/igerber/diff-diff/compare/v2.7.5...v2.7.6
[2.7.5]: https://github.com/igerber/diff-diff/compare/v2.7.4...v2.7.5
[2.7.4]: https://github.com/igerber/diff-diff/compare/v2.7.3...v2.7.4
[2.7.3]: https://github.com/igerber/diff-diff/compare/v2.7.2...v2.7.3
[2.7.2]: https://github.com/igerber/diff-diff/compare/v2.7.1...v2.7.2
[2.7.1]: https://github.com/igerber/diff-diff/compare/v2.7.0...v2.7.1
[2.7.0]: https://github.com/igerber/diff-diff/compare/v2.6.1...v2.7.0
[2.6.1]: https://github.com/igerber/diff-diff/compare/v2.6.0...v2.6.1
[2.6.0]: https://github.com/igerber/diff-diff/compare/v2.5.0...v2.6.0
[2.5.0]: https://github.com/igerber/diff-diff/compare/v2.4.3...v2.5.0
[2.4.3]: https://github.com/igerber/diff-diff/compare/v2.4.2...v2.4.3
[2.4.2]: https://github.com/igerber/diff-diff/compare/v2.4.1...v2.4.2
[2.4.1]: https://github.com/igerber/diff-diff/compare/v2.4.0...v2.4.1
[2.4.0]: https://github.com/igerber/diff-diff/compare/v2.3.2...v2.4.0
[2.3.2]: https://github.com/igerber/diff-diff/compare/v2.3.1...v2.3.2
[2.3.1]: https://github.com/igerber/diff-diff/compare/v2.3.0...v2.3.1
[2.3.0]: https://github.com/igerber/diff-diff/compare/v2.2.1...v2.3.0
[2.2.1]: https://github.com/igerber/diff-diff/compare/v2.2.0...v2.2.1
[2.2.0]: https://github.com/igerber/diff-diff/compare/v2.1.9...v2.2.0
[2.1.9]: https://github.com/igerber/diff-diff/compare/v2.1.8...v2.1.9
[2.1.8]: https://github.com/igerber/diff-diff/compare/v2.1.7...v2.1.8
[2.1.7]: https://github.com/igerber/diff-diff/compare/v2.1.6...v2.1.7
[2.1.6]: https://github.com/igerber/diff-diff/compare/v2.1.5...v2.1.6
[2.1.5]: https://github.com/igerber/diff-diff/compare/v2.1.4...v2.1.5
[2.1.4]: https://github.com/igerber/diff-diff/compare/v2.1.3...v2.1.4
[2.1.3]: https://github.com/igerber/diff-diff/compare/v2.1.2...v2.1.3
[2.1.2]: https://github.com/igerber/diff-diff/compare/v2.1.1...v2.1.2
[2.1.1]: https://github.com/igerber/diff-diff/compare/v2.1.0...v2.1.1
[2.1.0]: https://github.com/igerber/diff-diff/compare/v2.0.3...v2.1.0
[3.3.2]: https://github.com/igerber/diff-diff/compare/v3.3.1...v3.3.2
[3.3.1]: https://github.com/igerber/diff-diff/compare/v3.3.0...v3.3.1
[3.3.0]: https://github.com/igerber/diff-diff/compare/v3.2.0...v3.3.0
[3.2.0]: https://github.com/igerber/diff-diff/compare/v3.1.3...v3.2.0
[3.1.3]: https://github.com/igerber/diff-diff/compare/v3.1.2...v3.1.3
[3.1.2]: https://github.com/igerber/diff-diff/compare/v3.1.1...v3.1.2
[3.1.1]: https://github.com/igerber/diff-diff/compare/v3.1.0...v3.1.1
[3.1.0]: https://github.com/igerber/diff-diff/compare/v3.0.2...v3.1.0
[3.0.2]: https://github.com/igerber/diff-diff/compare/v3.0.1...v3.0.2
[2.0.3]: https://github.com/igerber/diff-diff/compare/v2.0.2...v2.0.3
[2.0.2]: https://github.com/igerber/diff-diff/compare/v2.0.1...v2.0.2
[2.0.1]: https://github.com/igerber/diff-diff/compare/v2.0.0...v2.0.1
[2.0.0]: https://github.com/igerber/diff-diff/compare/v1.4.0...v2.0.0
[1.4.0]: https://github.com/igerber/diff-diff/compare/v1.3.1...v1.4.0
[1.3.1]: https://github.com/igerber/diff-diff/compare/v1.3.0...v1.3.1
[1.3.0]: https://github.com/igerber/diff-diff/compare/v1.2.1...v1.3.0
[1.2.1]: https://github.com/igerber/diff-diff/compare/v1.2.0...v1.2.1
[1.2.0]: https://github.com/igerber/diff-diff/compare/v1.1.1...v1.2.0
[1.1.1]: https://github.com/igerber/diff-diff/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/igerber/diff-diff/compare/v1.0.2...v1.1.0
[1.0.2]: https://github.com/igerber/diff-diff/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/igerber/diff-diff/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/igerber/diff-diff/compare/v0.6.0...v1.0.0
[0.6.0]: https://github.com/igerber/diff-diff/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/igerber/diff-diff/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/igerber/diff-diff/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/igerber/diff-diff/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/igerber/diff-diff/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/igerber/diff-diff/releases/tag/v0.1.0
