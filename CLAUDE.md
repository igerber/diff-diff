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

## Testing Conventions

- **`ci_params` fixture** (session-scoped in `conftest.py`): Use `ci_params.bootstrap(n)` and
  `ci_params.grid(values)` to scale iterations in pure Python mode. For SE convergence tests,
  use `ci_params.bootstrap(n, min_n=199)` with conditional tolerance:
  `threshold = 0.40 if n_boot < 100 else 0.15`.
- **`assert_nan_inference()`** from conftest.py: Use to validate ALL inference fields are
  NaN-consistent. Don't check individual fields separately.
- **Slow test suites**: `tests/test_trop.py` is very time-consuming. Skip with
  `pytest --ignore=tests/test_trop.py` for unrelated changes.
- **Behavioral assertions**: Always assert expected outcomes, not just no-exception.
  Bad: `result = func(bad_input)`. Good: `result = func(bad_input); assert np.isnan(result.coef)`.

## Key Reference Files

| File | Contains |
|------|----------|
| `docs/methodology/REGISTRY.md` | Academic foundations, equations, edge cases — **consult before methodology changes** |
| `CONTRIBUTING.md` | Documentation requirements, test writing guidelines |
| `.claude/commands/dev-checklists.md` | Checklists for params, methodology, warnings, reviews, bugs (run `/dev-checklists`) |
| `.claude/memory.md` | Debugging patterns, tolerances, API conventions (git-tracked) |
| `docs/performance-plan.md` | Performance optimization details |
| `docs/benchmarks.rst` | Validation results vs R |

## Workflow

- For non-trivial tasks, use `EnterPlanMode`. Consult `docs/methodology/REGISTRY.md` for methodology changes.
- For bug fixes, grep for the pattern across all files before fixing.
- Follow the relevant development checklists (run `/dev-checklists`).
- Before submitting: run `/pre-merge-check`.
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
timestamp, verdict, issue counts). Update sentinel. Collect feedback and revise if needed.
Touch review file after revision to avoid staleness check failure.

**If skipped**: Write a minimal review marker to `~/.claude/plans/<plan-basename>.review.md`:
```yaml
---
plan: <plan-file-path>
reviewed_at: <ISO 8601 timestamp>
verdict: "Skipped"
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
