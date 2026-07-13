# diff-diff Roadmap

Forward-looking plan for diff-diff, organized as queued work, candidates under consideration, and longer-term directions. For what exists today, see the [README](README.md) estimator catalog and the [API reference on Read the Docs](https://diff-diff.readthedocs.io); for shipped history and release notes, see [CHANGELOG.md](CHANGELOG.md).

---

## Shipping Next

Queued work, ordered by expected leverage. Each item is its own PR. Ordering is priority-sequenced, not time-committed.

### Practitioner-ready output

- **Context-aware `practitioner_next_steps()`.** Substitutes actual column names from fitted results instead of generic placeholders, so next-step guidance is executable rather than illustrative. (Standalone follow-up to the `BusinessReport` / `DiagnosticReport` layer; tracked under the AI-Agent Track too.)

### Practitioner tutorials

- **dCDH comprehensive tutorial.** One notebook covering reversible treatment, dynamic event study, covariates, trends, HonestDiD on placebos, and survey. Favara-Imbs (2015) banking-deregulation replication as the headline application.
- **BRFSS repeated-cross-section tutorial.** State-policy DiD replication using `CallawaySantAnna(panel=False)` with design-based SEs and HonestDiD sensitivity. Targets the highest-demand survey-DiD audience segment.
- **Marketing Campaign Lift tutorial** (CallawaySantAnna, staggered geo rollout).
- **Pricing / Promotion Impact tutorial** (ContinuousDiD dose-response).

### Survey breadth and validation

- **Two-phase sampling + multi-stage cluster R-validation tests.** Extend existing survey cross-validation to NHANES two-phase design and MICS/DHS/NCVS multi-stage cluster. Closes a practitioner-design gap and firms up the design-based variance claim.

---

## Under Consideration

Research-informed candidates. Each has a rationale, a tractability note, and a commit criterion. Papers are academic references, so citation is fine.

### Methodology extensions

- **Nonparametric / flexible outcome regression for `EfficientDiD` DR covariate path** (Chen, Sant'Anna & Xie, arXiv:2506.17729, 2025, Section 4). The shipped staggered `EfficientDiD` uses a linear OLS outcome regression in its doubly-robust covariate path; that preserves DR consistency but does not generically attain the semiparametric efficiency bound unless the conditional mean is linear in the covariates. Replacing the OLS outcome regression with sieve / kernel / ML nuisance estimation (as the paper's Section 4 allows) would close the efficiency gap on the covariate path. Tractability: medium; the hook points are in `diff_diff/efficient_did_covariates.py`. **Commit when**: a paper-review synthesis is written, with an implementation plan for the nonparametric OR that preserves the existing DR consistency guarantees and survey-weighted variance surface.
- **Distributional DiD for staggered timing** (Ciaccio, arXiv:2408.01208, 2024). New estimator extending Callaway-Li QTT to staggered adoption. `CallawaySantAnna` currently gives mean ATT only; this unlocks quantile effects. Tractability: medium. **Reviewed 2026-07** (`docs/methodology/papers/ciaccio-2024-review.md`); implementation deferred pending demand. **Commit when**: a health-econ or public-health user reports need for quantile effects in a repeated-cross-section design.
- **Local Projections DiD** (Dube, Girardi, Jordà & Taylor, JAE 2025). New estimator with flexible impulse-response and robustness to dynamic misspecification; natural for anticipation-prone settings. Tractability: well-scoped. **Commit when**: a methodology review confirms the dynamic variant's variance derivation fits our SE helpers.
- **Few-treated-units inference option** (Alvarez, Ferman & Wüthrich, arXiv:2504.19841, 2025). `inference=` option covering t(G-1) corrections, randomization inference, and Ferman-Pinto-style permutation tests. Current SE paths assume large-G asymptotics. Tractability: medium. **Commit when**: a user reports sparse-treatment pain.
- **Riesz-representation sensitivity** (Bach et al., arXiv:2510.09064, 2025). Confounder-based sensitivity bound complementing HonestDiD's trend-based bound. Tractability: medium. **Commit when**: HonestDiD users ask for confounder bounds.
- **Compositional-change inference** (Sant'Anna & Xu, arXiv:2304.13925 v3, 2025). Corrects inference for rolling-panel repeated-cross-section designs (ACS, CPS) where sample composition changes across periods. Tractability: medium. **Commit when**: BRFSS tutorial or an applied user surfaces the issue.

### Post-estimation and export capabilities

Framed as what diff-diff offers, not which external tool plugs in:

- **Standard post-estimation interface.** Expose `.predict()` and `.vcov()` in shapes that common post-estimation slope / contrast / hypothesis-test interfaces consume. Tractability: small Protocol addition plus compatibility shim. **Commit when**: a concrete contract with one of the existing results objects is defined.
- **Publication-table export.** `result.to_table()` producing publication-quality HTML / PNG / LaTeX tables via an optional extra. Tractability: low. **Commit when**: `BusinessReport` ships so the formatter can piggyback on its summary pipeline.
- **Survey design object interop.** `SurveyDesign.from_design_object(...)` / `.to_design_object(...)` for accepting and emitting standard Python survey-design objects. Tractability: depends on upstream API stability. **Commit when**: a stable public design surface exists upstream.
- **Pluggable regression engine for TWFE / event-study paths.** Opt-in `engine=` parameter allowing alternative backends. Tractability: contained change plus coefficient-parity CI. **Commit when**: profiling shows material wins on real practitioner panels.

### Parked (explicit non-goals)

- New estimators beyond the list above without a user-driven demand signal.
- Calibration / raking / post-stratification as first-party features (remain upstream; document the handoff).
- Product Launch Regional Rollout and Loyalty Program tutorials (defer until a practitioner request).
- Methodology-vs-alternative comparison pages (replaced by BusinessReport and the tutorials that showcase diff-diff's output directly).

---

## AI-Agent Track

Long-running program, framed as "building toward" rather than with discrete ship dates.

**Vision.** A practitioner hands an AI agent a business scenario. The agent, with diff-diff as its toolkit, interprets the scenario, selects the correct estimator and identification strategy, executes the analysis with correct diagnostics and sensitivity, and returns a business-ready report. Practitioners never see raw coefficients unless they want to.

**Building blocks already in place.** Several agent-facing building blocks already ship - Baker et al. (2025) 8-step workflow enforcement, runtime LLM guides via `get_llm_guide(...)`, `profile_panel(...)` structural panel profiling, an "For AI agents" package-docstring entry block, and silent-operation warnings. See the README "For AI Agents" section and the bundled `llms*.txt` guides for the current surface.

**Next blocks toward the vision.**

- **Structured `sanity_checks` block in BR/DR** - machine-legible pass / warn / fail signals for pretrends, power, forbidden-comparisons, event-study cleanliness, placebo, and sensitivity, so agents dispatch on a stable schema rather than parsing prose. Highest-leverage net-new agent decision surface; orthogonal to existing `caveats` and to fit-time validators.
- **Post-hoc mismatch detection in BR/DR output** - originally proposed as Wave 2 but rescoped after a plan review showed most candidate checks duplicate fit-time validators (which raise `ValueError` before any fitted result exists) or the existing `caveats` block (TWFE-on-staggered is already surfaced via `bacon_contamination`). Held for revisiting only if the `sanity_checks` rollout uncovers genuine post-fit mismatch signals not caught by current surfaces.
- **Context-aware `practitioner_next_steps()`** that substitutes actual column names - turns guidance into executable recommendations.
- **Unified `assess_*` verb** across estimator native-diagnostic methods for a single discoverable convention.
- **End-to-end scenario walkthrough templates** - reusable orchestration recipes an agent can adapt from data ingest through business-ready output.

---

## Long-term Research Directions

Frontier methods that may graduate to Under Consideration given time and research signals.

### Causal Duration Analysis with DiD

Extends DiD to duration / survival outcomes where standard methods fail (hazard rates, time-to-event). Duration analogue of parallel trends; avoids distributional and hazard-function assumptions.

**Reference**: Deaner & Ku (2025), *AEA Conference Paper*.

### CATT Meta-Learner for Heterogeneous Effects

ML-powered conditional ATT, using a doubly robust meta-learner to discover which units benefit most from treatment.

**Reference**: Lan, Chang, Dillon & Syrgkanis (2025), working paper.

### Causal Forests for DiD

Machine-learning methods for discovering heterogeneous treatment effects in DiD settings. Recent applied-econometrics work (Gavrilova et al. 2025, *Journal of Applied Econometrics*) demonstrates the approach on panel data.

**References**: Athey & Wager (2019), *Annals of Statistics*; Kattenberg, Scheer & Thiel (2023), *CPB Discussion Paper*.

### Matrix Completion Methods

Unified framework encompassing synthetic control and regression approaches via low-rank matrix recovery.

**Reference**: Athey et al. (2021), *Journal of the American Statistical Association*.

### Double / Debiased ML for DiD

Machine learning nuisance estimation in high-dimensional DiD settings.

**Reference**: Chernozhukov et al. (2018), *The Econometrics Journal*.

### Alternative Inference Methods

- **Randomization inference**: exact p-values for small samples.
- **Bayesian DiD**: priors on parallel-trends violations.
- **Conformal inference**: prediction intervals with finite-sample guarantees.

---

## Contributing

Interested in contributing? Under Consideration items with clear commit criteria are good candidates. See the [GitHub repository](https://github.com/igerber/diff-diff) for open issues.

Key references:

- [Roth et al. (2023)](https://www.sciencedirect.com/science/article/abs/pii/S0304407623001318). "What's Trending in Difference-in-Differences?" *Journal of Econometrics*.
- [Baker et al. (2025)](https://arxiv.org/pdf/2503.13323). "Difference-in-Differences Designs: A Practitioner's Guide."
- [Abadie, Angrist, Frandsen & Pischke (2025)](https://www.nber.org/papers/w34550). "Harvesting Differences-in-Differences and Event-Study Evidence." NBER WP 34550.
