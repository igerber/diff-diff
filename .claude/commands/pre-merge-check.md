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

#### 2.1 Inference & Parameter Pattern Checks (for methodology files)

> **Canonical definitions** — This section is referenced by `/submit-pr` and `/push-pr-update`. Keep it as the single source of truth for methodology pattern checks.

If any methodology files changed, run these pattern checks on the **changed methodology files only**:

**Check A — Inline inference computation**:
```bash
grep -n "t_stat[[:space:]]*=[[:space:]]*[^#]*/ *se" <changed-methodology-files> | grep -v "safe_inference"
```
Flag each match: "Consider using `safe_inference()` from `diff_diff.utils` instead of inline t_stat computation."

**Check B — Zero-SE fallback to 0.0 instead of NaN**:
```bash
grep -En "if.*(se|SE).*>.*0.*else[[:space:]]+(0\.0|0)" <changed-methodology-files>
```
Flag each match: "SE=0 should produce NaN, not 0.0, for inference fields."

**Check C — New `self.X` in `__init__` not in `get_params()`**:
```bash
# Extract new self.X assignments from diff
git diff HEAD -- <changed-methodology-files> | grep "^+" | grep "self\.\w\+ = \w\+" | sed 's/.*self\.\(\w\+\).*/\1/' | sort -u
# For each extracted param name, check if it appears in get_params()
grep "get_params" <changed-methodology-files> -A 30 | grep "<param_name>"
```
Flag missing params: "New parameter `X` stored in `__init__` but not found in `get_params()` return dict."

**Check D — `compute_confidence_interval` called without NaN guard**:
```bash
grep -n "compute_confidence_interval" <changed-methodology-files> | grep -v "safe_inference\|if.*isfinite\|if.*se.*>"
```
Flag each match: "Verify CI computation handles non-finite SE (use `safe_inference()` or add guard)."

**Report findings**: If matches found, list each with file:line and the recommended fix.

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

#### 2.4 Docstring Staleness Check (for changed .py files)

For functions with changed signatures, verify their docstrings are up to date:

```bash
# Find functions with changed signatures
git diff HEAD -- <changed-py-files> | grep "^+.*def " | head -10
```

For each changed function, flag: "Verify docstring Parameters section matches updated signature for: `<function_name>`"

#### 2.5 Methodology Documentation Check

If any methodology files changed, check whether `docs/methodology/REGISTRY.md` was also
modified in the changed file set (from Section 1).

If methodology files changed but REGISTRY.md was NOT modified, flag:
"Methodology files changed but `docs/methodology/REGISTRY.md` was not updated. If your
changes deviate from reference implementations, document them using a reviewer-recognized
label (`**Note:**`, `**Deviation from R:**`, or `**Note (deviation from R):**`) —
undocumented deviations are flagged as P1 by the AI reviewer and cannot be mitigated
by TODO.md."

This is a WARNING, not a blocker — not every methodology change involves a deviation.

#### 2.6 Secret Scanning Patterns (Canonical Definitions)

> These patterns are referenced by `/submit-pr` and `/push-pr-update`.

**Content pattern** (use with `-G` flag, `--name-only` to avoid leaking secrets):
```bash
-G "(AKIA[A-Z0-9]{16}|ghp_[a-zA-Z0-9]{36}|sk-[a-zA-Z0-9]{48}|gho_[a-zA-Z0-9]{36}|[Aa][Pp][Ii][_-]?[Kk][Ee][Yy][[:space:]]*[=:]|[Ss][Ee][Cc][Rr][Ee][Tt][_-]?[Kk][Ee][Yy][[:space:]]*[=:]|[Pp][Aa][Ss][Ss][Ww][Oo][Rr][Dd][[:space:]]*[=:]|[Pp][Rr][Ii][Vv][Aa][Tt][Ee][_-]?[Kk][Ee][Yy]|[Bb][Ee][Aa][Rr][Ee][Rr][[:space:]]+[a-zA-Z0-9_-]+|[Tt][Oo][Kk][Ee][Nn][[:space:]]*[=:])"
```

**Sensitive filename pattern**:
```bash
grep -iE "(\.env|credentials|secret|\.pem|\.key|\.p12|\.pfx|id_rsa|id_ed25519)$"
```

**Usage**: Apply content pattern to `--cached` for staged changes, or `<ref>..HEAD` for already-committed changes. Always use `--name-only` and `|| true`.

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

### Methodology Deviation Documentation
- [ ] If deviating from reference implementation: added a reviewer-recognized label
      (`**Note:**`, `**Deviation from R:**`, or `**Note (deviation from R):**`) in REGISTRY.md
- [ ] No undocumented methodology deviations (AI reviewer flags these as P1)
```

#### If Documentation Files Changed
```
### Documentation Sync
- [ ] Docstrings updated for changed function signatures
- [ ] README updated if user-facing behavior changes
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
