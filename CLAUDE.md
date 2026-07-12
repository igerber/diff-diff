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
`pyproject.toml`; the `ruff`/`black` pins are mirrored in
`.github/workflows/lint.yml` (the ungated `Lint Gate` check runs `ruff check`
+ `black --check` on every PR push; sync enforced by `TestLintWorkflowPinSync`),
while `mypy` is pinned in `pyproject.toml` only until the type-check CI job
lands — update both surfaces together on any bump. Refresh local tools with `pip install -e ".[dev]"`
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
   DifferenceInDifferences (base class)
   ├── TwoWayFixedEffects (inherits get_params/set_params)
   └── MultiPeriodDiD (inherits get_params/set_params)

   Standalone estimators (each has own get_params/set_params):
   ├── CallawaySantAnna
   ├── SunAbraham
   ├── ImputationDiD
   ├── TwoStageDiD
   ├── TripleDifference
   ├── TROP
   ├── StackedDiD
   ├── SyntheticDiD
   └── BaconDecomposition
   ```
   When adding params to `DifferenceInDifferences.get_params()`, subclasses inherit automatically.
   Standalone estimators must be updated individually.
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

**TODO.md format** — for deferring P2/P3 items only (P0/P1 cannot be deferred):

Add a row to `TODO.md`. If the item is genuinely shippable (clear path, no external
blocker), put it under **Actionable Backlog** in the appropriate sub-section
(`Methodology / correctness`, `Performance`, or `Testing / docs`). If it is blocked, put it
under **Deferred / Documented** in the matching blocker sub-section (`Paper-gated / needs
methodology derivation`, `Needs external reference (R / Stata / Julia)`, `Parked — pending
user demand / out of scope`, or `Won't-fix / waived`). Either way the AI reviewer's
deviation-grep resolves on the row's `Location` + reason text. The two buckets use
different table shapes — Actionable rows carry an `Effort` column, Deferred rows a `PR`
column:

Actionable Backlog:

| Issue | Location | Origin | Effort | Priority |
|-------|----------|--------|--------|----------|
| Description of the work item | `file.py` | #NNN | Quick/Mid/Heavy | Medium/Low |

Deferred / Documented:

| Issue | Location | PR | Priority |
|-------|----------|----|----------|
| Description of deferred item | `file.py` | #NNN | Medium/Low |

## README discipline

`README.md` is a **landing page**, not the documentation. Target ~190 lines. The 3,119-line README that existed before the 2026-04 docs refresh grew because workflow conventions told contributors to add to README on every change.

When adding new functionality, the source of truth is:

- **`diff_diff/guides/llms.txt`** for the AI-agent contract (one-line catalog entry per estimator with paper citation + RTD link). This file is bundled in the wheel and published on RTD via `docs/conf.py` `html_extra_path`.
- **`docs/api/*.rst`** for full API reference.
- **`docs/references.rst`** for scholarly citations.
- **`docs/tutorials/*.ipynb`** for hands-on examples.
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
| `docs/doc-deps.yaml` | Source-to-documentation dependency map — **consult when any source file changes** |
| `CONTRIBUTING.md` | Documentation requirements, test writing guidelines |
| `.claude/commands/dev-checklists.md` | Checklists for params, methodology, warnings, reviews, bugs (run `/dev-checklists`) |
| `.claude/memory.md` | Debugging patterns, tolerances, API conventions (git-tracked) |
| `diff_diff/guides/llms-practitioner.txt` | Baker et al. (2025) 8-step practitioner workflow for AI agents (accessible at runtime via `diff_diff.get_llm_guide("practitioner")`) |
| `docs/performance-plan.md` | Performance optimization details |
| `docs/benchmarks.rst` | Validation results vs R |

## Workflow

- CI tests are gated behind the `ready-for-ci` label. The `CI Gate` required status check
  enforces this — PRs cannot merge until the label is added. Tests run automatically once
  the label is present.
- For non-trivial tasks, use `EnterPlanMode`. Consult `docs/methodology/REGISTRY.md` for methodology changes.
- When modifying source files in `diff_diff/`, consult `docs/doc-deps.yaml` to identify impacted documentation. Run `/docs-impact` to see the full list.
- For bug fixes, grep for the pattern across all files before fixing.
- Follow the relevant development checklists (run `/dev-checklists`).
- Before submitting: run `/pre-merge-check`, then `/ai-review-local` for pre-PR AI review.
- Submit with `/submit-pr`.

## Plan Review Before Approval

When writing a new plan file (via EnterPlanMode), update the sentinel:
```bash
echo "<plan-file-path>" > ~/.claude/plans/.last-reviewed
```

Before calling `ExitPlanMode`, offer the user an independent plan review via `AskUserQuestion`:
- "Run review agent for independent feedback" (Recommended)
- "Present plan for approval as-is"

**If review requested**: Spawn review agent (Task tool, `subagent_type: "general-purpose"`)
to read `.claude/commands/review-plan.md` and follow Steps 2-5. Display output in conversation.
Save to `~/.claude/plans/<plan-basename>.review.md` with YAML frontmatter (plan path,
timestamp, assessment, issue counts). Update sentinel. Collect feedback and revise if needed.
Touch review file after revision to avoid staleness check failure.

**If skipped**: Write a minimal review marker to `~/.claude/plans/<plan-basename>.review.md`:
```yaml
---
plan: <plan-file-path>
reviewed_at: <ISO 8601 timestamp>
assessment: "Skipped"
critical_count: 0
medium_count: 0
low_count: 0
flags: []
---
Review skipped by user.
```
Update sentinel. The `check-plan-review.sh` hook enforces this workflow.

**Rollback**: To remove the plan review workflow, delete this section from CLAUDE.md,
remove the `PreToolUse` entry from `.claude/settings.json`, and delete
`.claude/hooks/check-plan-review.sh`.
