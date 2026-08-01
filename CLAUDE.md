# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

diff-diff is a Python library for Difference-in-Differences (DiD) causal inference analysis. It provides sklearn-like estimators with statsmodels-style output for econometric analysis.

## Common Commands

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run a specific test file
pytest tests/test_estimators.py

# Run a specific test
pytest tests/test_estimators.py::TestDifferenceInDifferences::test_basic_did

# Format code
black diff_diff tests

# Lint code
ruff check diff_diff tests

# Type checking
mypy diff_diff
```

Lint/format/type tool versions are **pinned exactly** in the `dev` extra of
`pyproject.toml` and mirrored in `.github/workflows/lint.yml` (the ungated
`Lint Gate` check runs `ruff check` + `black --check` + `mypy diff_diff` on
every PR push; sync enforced by `TestLintWorkflowPinSync`) — update both
surfaces together on any bump. Mypy is enforced at ZERO errors; new code
must type-check cleanly. Refresh local tools with `pip install -e ".[dev]"`
(the pinned tools need Python >= 3.10; the library floor stays 3.9). Some
ruff rules are deliberately ignored per-file (`[tool.ruff.lint.per-file-ignores]`)
— don't "fix" those patterns. One-time setup so `git blame` skips the 2026-07
bulk-normalization commits: `git config blame.ignoreRevsFile .git-blame-ignore-revs`.

### Rust Backend Commands

```bash
# Build Rust backend for development (requires Rust toolchain)
maturin develop

# Build with release optimizations
maturin develop --release

# Build with platform BLAS (macOS — links Apple Accelerate)
maturin develop --release --features accelerate

# Build with platform BLAS (Linux — requires libopenblas-dev)
maturin develop --release --features openblas

# Build without BLAS (Windows, or explicit pure Rust)
maturin develop --release

# Force pure Python mode (disable Rust backend)
DIFF_DIFF_BACKEND=python pytest

# Force Rust mode (fail if Rust not available)
DIFF_DIFF_BACKEND=rust pytest

# Run Rust backend equivalence tests
pytest tests/test_rust_backend.py -v
```

## Key Design Patterns

1. **sklearn-like API**: Estimators use `fit()` method, `get_params()`/`set_params()` for configuration
2. **Formula interface**: Supports R-style formulas like `"outcome ~ treated * post"`
3. **Fixed effects handling**:
   - `fixed_effects` parameter creates dummy variables (for low-dimensional FE)
   - `absorb` parameter uses within-transformation (for high-dimensional FE)
4. **Results objects**: Rich dataclass containers with `summary()`, `to_dict()`, `to_dataframe()`
5. **Unified `linalg.py` backend**: ALL estimators use `solve_ols()` / `compute_robust_vcov()`
6. **Inference computation**: ALL inference fields (t_stat, p_value, conf_int) MUST be computed
   together using `safe_inference()` from `diff_diff.utils`. Never compute individually.
7. **Estimator inheritance** — understanding this prevents consistency bugs:
   ```
   BaseEstimator (diff_diff/_base.py — shared get_params/set_params mixin)
   ├── DifferenceInDifferences
   │   ├── TwoWayFixedEffects
   │   ├── MultiPeriodDiD
   │   └── SyntheticDiD
   └── every other estimator class (CallawaySantAnna, SunAbraham,
       ImputationDiD, TwoStageDiD, TripleDifference, TROP, StackedDiD,
       BaconDecomposition, ... — 25 classes total)
   ```
   `get_params` introspects the `__init__` signature and `set_params` is
   TRANSACTIONAL via probe re-init (`type(self)(**merged)` validates before
   any mutation) — so a new `__init__` param is automatically in
   `get_params`/`set_params` for every class, and `set_params` enforces
   exactly the constructor's validation, eagerly. Per-class accommodations
   are declarative class attrs (`_PARAM_ATTR_ALIASES`,
   `_DERIVED_CONFIG_ATTRS`, `_normalize_set_params`); the cross-estimator
   contract is pinned by `tests/test_base_estimator.py` (dynamic roster —
   new estimators are automatically enrolled and must mix BaseEstimator in).
8. **Dependencies**: numpy, pandas, and scipy ONLY. No statsmodels.

## Documenting Deviations (AI Review Compatibility)

The AI PR reviewer recognizes deviations as documented (and downgrades them to P3) ONLY
when they use specific label patterns in `docs/methodology/REGISTRY.md`. Using different
wording will cause a P1 finding ("undocumented methodology deviation").

**Recognized REGISTRY.md labels** — use one of these in the relevant estimator section:

| Label | When to use | Example |
|-------|------------|---------|
| `- **Note:** <text>` | Defensive enhancements, implementation choices | `- **Note:** Defensive enhancement matching CallawaySantAnna NaN convention` |
| `- **Deviation from R:** <text>` | Intentional differences from R packages | `- **Deviation from R:** R's fixest uses t-distribution at all levels` |
| `**Note (deviation from R):** <text>` | Combined form, inline within edge case bullets | See SyntheticDiD section in REGISTRY.md |

**Tracking-file map** — for deferring P2/P3 items only (P0/P1 cannot be deferred):

- **Shippable** (clear path, no external blocker) → a row in `TODO.md` under
  **Actionable Backlog**, in the matching sub-section (`Methodology / correctness`,
  `Performance`, or `Testing / docs`).
- **Blocked** → a row in `DEFERRED.md` under the matching blocker section
  (`Paper-gated / needs methodology derivation`, `Needs external reference
  (R / Stata / Julia)`, `Parked — pending user demand / out of scope`, or
  `Version-gated (v4)`).
- **Decisions** (won't-fix / waived): if the decision pins **user-visible behavior or
  methodology**, record it as a REGISTRY.md Note using the labels above; if it is
  **internal engineering** (refactor waiver, perf trade-off, test-infrastructure call),
  add it to `DEFERRED.md` → **Decision record — won't-fix / waived**.
- **Version-gated lifecycle items** (deprecated-kwarg removals, v4 default flips):
  `docs/v4-deprecations.yaml` (CI-enforced) is the lifecycle authority — never restate
  ledger status/targets in a row. A row carrying real implementation work (e.g. a soak
  or recapture protocol) may exist in TODO.md/DEFERRED.md but must cross-link its
  `M-xxx` id.
- **Monitoring / current-state notes** (module sizes, tooling posture, platform quirks)
  go in `docs/dev-status.md`, not a backlog row.

The AI reviewer's deviation-grep resolves on a row's `Location` + reason text in EITHER
`TODO.md` or `DEFERRED.md`. The two files use different table shapes — Actionable rows
carry an `Effort` column, DEFERRED rows a `PR` column:

TODO.md → Actionable Backlog:

| Issue | Location | Origin | Effort | Priority |
|-------|----------|--------|--------|----------|
| Description of the work item | `file.py` | #NNN | Quick/Mid/Heavy | Medium/Low |

DEFERRED.md (blocker sections):

| Issue | Location | PR | Priority |
|-------|----------|----|----------|
| Description of deferred item | `file.py` | #NNN | Medium/Low |

## README discipline

`README.md` is a **landing page**, not the documentation. Target ~190 lines. The 3,119-line README that existed before the 2026-04 docs refresh grew because workflow conventions told contributors to add to README on every change.

When adding new functionality, the source of truth is:

- **`diff_diff/guides/llms.txt`** for the AI-agent contract (one-line catalog entry per estimator with paper citation + RTD link). This file is bundled in the wheel and published on RTD via `docs/conf.py` `html_extra_path`.
- **`docs/api/*.rst`** for full API reference.
- **`docs/references.rst`** for scholarly citations.
- **`docs/tutorials/*.ipynb`** for hands-on examples. New notebooks are registered in
  `docs/tutorials/index.rst` (toctree entry with a short display label + a card in the
  matching group), NOT in `docs/index.rst` - the root toctree lists only the 5 section
  landing pages so the navbar stays at 5 links. These IA invariants (plus homepage
  estimator-table parity with the API catalog, and the rule that any class documented
  with `:no-index:` on a module page keeps a canonical autosummary entry in
  `docs/api/index.rst`) are CI-enforced by `tests/test_docs_ia.py`; the full list is in
  CONTRIBUTING.md "Docs IA invariants".
- **`CHANGELOG.md`** for release notes.
- **`README.md`** for ONE LINE in the `## Estimators` flat catalog (or `## Diagnostics & Sensitivity` for diagnostic-class features). Do NOT add usage examples, parameter tables, per-estimator sections, or full bibliographies.

`/docs-impact` and `/docs-check` enforce these surfaces. See `CONTRIBUTING.md` "README is a landing page, not the docs" for the full convention.

## Testing Conventions

- **`ci_params` fixture** (session-scoped in `conftest.py`): Use `ci_params.bootstrap(n)` and
  `ci_params.grid(values)` to scale iterations in pure Python mode. For SE convergence tests,
  use `ci_params.bootstrap(n, min_n=199)` with conditional tolerance:
  `threshold = 0.40 if n_boot < 100 else 0.15`.
- **`assert_nan_inference()`** from conftest.py: Use to validate ALL inference fields are
  NaN-consistent. Don't check individual fields separately.
- **Slow tests**: TROP methodology/global-method tests, Sun-Abraham bootstrap, and
  TROP-parity tests are marked `@pytest.mark.slow` and excluded by default via `addopts`.
  `test_trop.py` uses per-class markers (not file-level) so that validation, API, and
  solver tests still run in the pure Python CI fallback. Run `pytest -m ''` to include
  slow tests, or `pytest -m slow` to run only slow tests.
- **Behavioral assertions**: Always assert expected outcomes, not just no-exception.
  Bad: `result = func(bad_input)`. Good: `result = func(bad_input); assert np.isnan(result.coef)`.

## Key Reference Files

| File | Contains |
|------|----------|
| `docs/methodology/REGISTRY.md` | Academic foundations, equations, edge cases — **consult before methodology changes** |
| `docs/v4-design.md` + `docs/v4-deprecations.yaml` | 4.0 program design spec + CI-enforced deprecation ledger — **consult before any 4.0-program PR**; deviations must edit both in the same diff |
| `docs/doc-deps.yaml` | Source-to-documentation dependency map — **consult when any source file changes** |
| `CONTRIBUTING.md` | Documentation requirements, test writing guidelines, implementation guidelines |
| `.claude/memory.md` | Debugging patterns, tolerances, API conventions (git-tracked) |
| `diff_diff/guides/llms-practitioner.txt` | Baker et al. (2025) 8-step practitioner workflow for AI agents (accessible at runtime via `diff_diff.get_llm_guide("practitioner")`) |
| `docs/performance-plan.md` | Performance optimization details |
| `docs/benchmarks.rst` | Validation results vs R |

## Workflow

- CI tests are gated behind the `ready-for-ci` label. The `CI Gate` required status check
  enforces this — PRs cannot merge until the label is added. Tests run automatically once
  the label is present.
- To see what work is in flight, run `gh pr list --state open` and `git worktree list` — do
  not rely on a cached list of "active initiatives," which goes stale within hours. Open PRs,
  their branches, and worktrees are the source of truth.
- Do not create memories that record work *status* (which PRs merged, what's in progress,
  what's next) — it is derivable from git/gh above and goes stale immediately. Reserve memory
  for what git cannot tell you: a durable lesson learned and how to apply it; why an approach
  was rejected; external state (e.g. a paper submitted, awaiting response); or a decision
  pending on the user.
- For non-trivial tasks, use `EnterPlanMode`. Consult `docs/methodology/REGISTRY.md` for methodology changes.
- When modifying source files in `diff_diff/`, consult `docs/doc-deps.yaml` to identify impacted documentation. Run `/docs-impact` to see the full list.
- For bug fixes, grep for the pattern across all files before fixing, and fix every
  occurrence in the same PR (see CONTRIBUTING.md "Implementation Guidelines").
- Before submitting: run `/pre-merge-check`, then `/ai-review-local` for pre-PR AI review.
- Submit with `/submit-pr`.

## Plan Review Before Approval

The `check-plan-review.py` hook denies `ExitPlanMode` unless the plan's
review file (the helper-derived `review_path` — canonical basename + a
canonical-path digest, in `~/.claude/plans/`) exists with a `plan_sha256`
frontmatter field matching the SHA-256 of the plan file's CURRENT bytes. There
is no sentinel and no mtime check: any plan edit invalidates the review until
it is re-run or deliberately re-stamped.

Before calling `ExitPlanMode`, ALWAYS offer all three review options via
`AskUserQuestion`, with the `(Recommended)` tag chosen ADAPTIVELY from the
plan's complexity and risk (not a fixed default) — set exactly one:
- **Dual review** — the campaign-selected engine (two blind reviewers, Claude @
  Opus + codex `gpt-5.6-sol`, then merge/verify; Campaign 1 was exploratory /
  NON-GATING but showed dual catching 7/9 must-catch plan defects vs 1/9
  single). Recommend for SUBSTANTIVE / high-risk plans:
  estimator, methodology, variance/inference, or `docs/methodology/REGISTRY.md`
  changes; multi-file, architectural, or public-API changes.
- **Single review** (Opus only, no codex) — recommend for LOCALIZED, mechanical
  plans (a contained single-file change, test/doc tweaks) where dual's extra
  cost isn't warranted but a review still adds value.
- **Skip** — recommend only for TRIVIAL plans (typo, comment, obvious one-liner).

**If dual or single**: invoke the **`plan-review` skill**
(`.claude/skills/plan-review/`) in the chosen mode. Its Review phase snapshots
the plan via the tested helper (confirm the printed `plan_path` — the ingress
file is shared per-worktree), runs the reviewers (dual = Opus reviewer + codex
`gpt-5.6-sol` + merge/verify; single = the Opus reviewer alone), writes the
review to the helper-derived `review_path`, and runs `plan_snapshot.py persist`
(which re-verifies the live plan against the recorded snapshot — exit 3 = plan
changed mid-review, not persisted, re-review — stamps `plan:`/`plan_sha256:`
itself, and cleans up). In dual mode, if codex is unavailable the skill degrades
LOUDLY to a single-Claude review with a prominent warning (distinct from a
deliberate single-review choice). Display the review; collect feedback and
revise via the skill's Revise phase if needed — its re-review re-runs the SAME
chosen engine and writes the new `plan_sha256`; never re-stamp the old review's
hash onto content it did not examine. The hook denies on hash mismatch; `touch`
does nothing.

**If skip**: Write a minimal Skipped marker via the helper. First derive,
create, and PRINT the scratch dir in one Bash call — `SCRATCH="$(git rev-parse
--git-path plan-review)"; mkdir -p "$SCRATCH"; echo "$SCRATCH"` (on a fresh
worktree `.git/plan-review` does not exist yet, so the Write below would fail
without the `mkdir`; the `echo` gives you the literal path). Then, with the
Write tool (never echo/heredoc — the path is untrusted and never touches a
shell), write the raw plan path to the printed `<scratch>/plan-path.txt` (a
literal, not a `$SCRATCH` token — the Write tool does not expand variables), then
`python3 .claude/scripts/plan_snapshot.py snapshot --plan-path-file "$(git
rev-parse --git-path plan-review)/plan-path.txt"` (re-derived inline).
Confirm the printed `plan_path` is the plan you supplied. Write the Skipped meta
`{"reviewed_at": "<ISO 8601>", "assessment": "Skipped", "critical_count": 0,
"medium_count": 0, "low_count": 0, "flags": []}` to the printed `meta_path` and
`Review skipped by user.` to `body_path`, then `plan_snapshot.py persist
--state-file "<state_path>"` — the helper stamps the hash of the exact current
plan bytes. On a failure BEFORE persist (a bad meta/body Write) run plain
`plan_snapshot.py abort --state-file "<state_path>"`; do NOT abort after persist
— it self-cleans its own failures, so the snapshot is not retained either way.

**Rollback**: To remove the plan-review skill, delete
`.claude/skills/plan-review/`, restore `.claude/commands/review-plan.md` +
`revise-plan.md` from git history, and revert this section to spawn the single
review agent. To remove the gate entirely, also drop the `PreToolUse` entry from
`.claude/settings.json` and delete `.claude/hooks/check-plan-review.py`,
`.claude/scripts/plan_snapshot.py`, `tests/test_plan_review_hook.py`, and
`tests/test_plan_snapshot.py`.
