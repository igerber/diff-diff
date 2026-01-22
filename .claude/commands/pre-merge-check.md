---
description: Run pre-merge checks before submitting a PR
argument-hint: ""
---

# Pre-Merge Check

Run automated checks and display the pre-merge checklist before submitting a PR.

## Instructions

### 1. Identify Changed Files

Get the list of files that will be included in the PR:

```bash
# Get all changed files (tracked modifications + staged + untracked)
git diff --name-only HEAD
git diff --cached --name-only
git ls-files --others --exclude-standard
```

Categorize files into:
- **Methodology files**: `diff_diff/*.py` (excluding `__init__.py`)
- **Test files**: `tests/*.py`
- **Documentation files**: `*.md`, `*.rst`, `docs/**`

### 2. Run Automated Pattern Checks

#### 2.1 NaN Handling Pattern Check (for methodology files)

If any methodology files changed, run these pattern checks:

```bash
# Check for bad t-stat pattern (should use np.nan, not 0.0 or 0)
grep -n "if.*se.*>.*0.*else 0" diff_diff/*.py || true

# Check for potential division issues without NaN handling
grep -n "/ se\b" diff_diff/*.py | grep -v "np.nan" | grep -v "#" || true
```

**Report findings**: If matches found, warn about potential NaN handling issues.

#### 2.2 Test Existence Check

For each changed methodology file, check if corresponding test file was also changed:

| Methodology File | Expected Test File |
|------------------|-------------------|
| `diff_diff/staggered.py` | `tests/test_staggered.py` |
| `diff_diff/estimators.py` | `tests/test_estimators.py` |
| `diff_diff/twfe.py` | `tests/test_estimators.py` |
| `diff_diff/synthetic_did.py` | `tests/test_estimators.py` |
| `diff_diff/sun_abraham.py` | `tests/test_sun_abraham.py` |
| `diff_diff/triple_diff.py` | `tests/test_triple_diff.py` |
| `diff_diff/trop.py` | `tests/test_trop.py` |
| `diff_diff/bacon.py` | `tests/test_bacon.py` |
| `diff_diff/linalg.py` | `tests/test_linalg.py` |
| `diff_diff/utils.py` | `tests/test_utils.py` |
| `diff_diff/diagnostics.py` | `tests/test_diagnostics.py` |
| `diff_diff/prep.py` | `tests/test_prep.py` |
| `diff_diff/visualization.py` | `tests/test_visualization.py` |
| `diff_diff/honest_did.py` | `tests/test_honest_did.py` |
| `diff_diff/power.py` | `tests/test_power.py` |
| `diff_diff/pretrends.py` | `tests/test_pretrends.py` |

**Report**: List any methodology files without corresponding test changes.

#### 2.3 Docstring Check (heuristic)

For changed `.py` files, run a quick check for functions without docstrings:
```bash
# Find public function definitions (heuristic check)
grep -n "^def [^_]" <changed-py-files> | head -10
grep -n "^    def [^_]" <changed-py-files> | head -10
```

Note: This is a heuristic. Verify docstrings exist for new public functions.

### 3. Display Context-Specific Checklist

Based on what changed, display the appropriate checklist items:

#### Always Show (Core Checklist)
```
## Pre-Merge Checklist

Based on your changes to: <list of changed files>

### Behavioral Completeness
- [ ] Happy path tested
- [ ] Edge cases tested (empty data, NaN inputs, boundary conditions)
- [ ] Error/warning paths tested with behavioral assertions
```

#### If Methodology Files Changed
```
### Inference Field Consistency
- [ ] If SE can be 0/undefined, ALL inference fields (t-stat, p-value, CI) return NaN
- [ ] Aggregation methods propagate NaN correctly
- [ ] Bootstrap methods handle NaN in base estimates

### Control Group Logic (if adding new modes/code paths)
- [ ] Control group composition verified for new code paths
- [ ] "Not-yet-treated" excludes the treatment cohort itself
- [ ] Parameter interactions tested with all aggregation methods
```

#### If Documentation Files Changed
```
### Documentation Sync
- [ ] Docstrings updated for changed function signatures
- [ ] README updated if user-facing behavior changes
- [ ] REGISTRY.md updated if methodology edge cases change
```

#### If This Appears to Be a Bug Fix
```
### Pattern Consistency (Bug Fix)
- [ ] Grepped for similar patterns across codebase before fixing
- [ ] Fixed ALL occurrences, not just the one that was reported
- [ ] Verified fix doesn't break other code paths
```

### 4. Ask About Running Tests

Use AskUserQuestion to offer test options:

```
Would you like to run tests now?

Options:
1. Yes - run full test suite (pytest)
2. Yes - run only tests for changed files
3. No - skip tests for now
```

If user chooses option 1:
```bash
pytest
```

If user chooses option 2, run targeted tests based on changed files:
```bash
pytest tests/test_staggered.py tests/test_estimators.py  # (example)
```

### 5. Report Summary

```
## Pre-Merge Check Complete

### Automated Checks
- Pattern checks: [PASS/WARN - N potential issues found]
- Test coverage: [PASS/WARN - N methodology files without test changes]

### Manual Checklist
Review the checklist items above before running /submit-pr

### Findings to Address
<list any warnings from pattern checks>

### Next Steps
- Address any warnings above
- Complete manual checklist items
- When ready: /submit-pr "Your PR title"
```

## Notes

- This skill is read-only - it only analyzes and reports, doesn't modify files
- Run this BEFORE `/submit-pr` to catch issues early
- Pattern checks are heuristics - review flagged items manually to confirm
- If pattern checks find issues, fix them before submitting
