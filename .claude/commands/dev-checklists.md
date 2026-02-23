---
description: Development checklists for code changes (params, methodology, warnings, reviews, bugs)
argument-hint: "[checklist-name]"
---

# Development Checklists

## Adding a New Parameter to Estimators

When adding a new `__init__` parameter that should be available across estimators:

1. **Implementation** (for each affected estimator):
   - [ ] Add to `__init__` signature with default value
   - [ ] Store as `self.param_name`
   - [ ] Add to `get_params()` return dict
   - [ ] Handle in `set_params()` (usually automatic via `hasattr`)

2. **Consistency** - apply to all applicable estimators per the **Estimator inheritance** map in CLAUDE.md

3. **Testing**:
   - [ ] Test `get_params()` includes new param
   - [ ] Test parameter affects estimator behavior
   - [ ] Test with non-default value

4. **Downstream tracing**:
   - [ ] Before implementing: `grep -rn "self\.<param>" diff_diff/<module>.py` to find ALL downstream paths
   - [ ] Parameter handled in ALL aggregation methods (simple, event_study, group)
   - [ ] Parameter handled in bootstrap inference paths

5. **Documentation**:
   - [ ] Update docstring in all affected classes
   - [ ] Update CLAUDE.md if it's a key design pattern

## Implementing Methodology-Critical Code

When implementing or modifying code that affects statistical methodology (estimators, SE calculation, inference, edge case handling):

1. **Before coding - consult the Methodology Registry**:
   - [ ] Read the relevant estimator section in `docs/methodology/REGISTRY.md`
   - [ ] Identify the reference implementation(s) listed
   - [ ] Note the edge case handling requirements

2. **During implementation**:
   - [ ] Follow the documented equations and formulas
   - [ ] Match reference implementation behavior for standard cases
   - [ ] For edge cases: either match reference OR document deviation

3. **When deviating from reference implementations**:
   - [ ] Add a **Note** in the Methodology Registry explaining the deviation
   - [ ] Include rationale (e.g., "defensive enhancement", "R errors here")
   - [ ] Ensure the deviation is an improvement, not a bug

4. **Testing methodology-aligned behavior**:
   - [ ] Test that edge cases produce documented behavior (NaN, warning, etc.)
   - [ ] Assert warnings are raised (not just captured)
   - [ ] Assert the warned-about behavior actually occurred
   - [ ] For NaN results: assert `np.isnan()`, don't just check "no exception"
   - [ ] All inference fields computed via `safe_inference()` (not inline)

## Adding Warning/Error/Fallback Handling

When adding code that emits warnings or handles errors:

1. **Consult Methodology Registry first**:
   - [ ] Check if behavior is documented in edge cases section
   - [ ] If not documented, add it before implementing

2. **Verify behavior matches message**:
   - [ ] Manually trace the code path after warning/error
   - [ ] Confirm the stated behavior actually occurs

3. **Write behavioral tests**:
   - [ ] Don't just test "no exception raised"
   - [ ] Assert the expected outcome occurred
   - [ ] For fallbacks: verify fallback behavior was applied
   - [ ] Example: If warning says "setting NaN", assert `np.any(np.isnan(result))`

4. **Protect arithmetic operations**:
   - [ ] Wrap ALL related operations in `np.errstate()`, not just the final one
   - [ ] Include division, matrix multiplication, and any operation that can overflow/underflow

## Reviewing New Features or Code Paths

When reviewing PRs that add new features, modes, or code paths (learned from PR #97 analysis):

1. **Edge Case Coverage**:
   - [ ] Empty result sets (no matching data for a filter condition)
   - [ ] NaN/Inf propagation through ALL inference fields (SE, t-stat, p-value, CI)
   - [ ] Parameter interactions (e.g., new param x existing aggregation methods)
   - [ ] Control/comparison group composition for all code paths

2. **Documentation Completeness**:
   - [ ] All new parameters have docstrings with type, default, and description
   - [ ] Methodology docs match implementation behavior (equations, edge cases)
   - [ ] Edge cases documented in `docs/methodology/REGISTRY.md`

3. **Logic Audit for New Code Paths**:
   - [ ] When adding new modes (like `base_period="varying"`), trace ALL downstream effects
   - [ ] Check aggregation methods handle the new mode correctly
   - [ ] Check bootstrap/inference methods handle the new mode correctly
   - [ ] Explicitly test control group composition in new code paths

4. **Pattern Consistency**:
   - [ ] Search for similar patterns in codebase (e.g., `t_stat = x / se if se > 0 else ...`)
   - [ ] Ensure new code follows established patterns or updates ALL instances
   - [ ] If fixing a pattern, grep for ALL occurrences first:
     ```bash
     grep -n "if.*se.*> 0.*else" diff_diff/*.py
     ```

## Fixing Bugs Across Multiple Locations

When a bug fix involves a pattern that appears in multiple places (learned from PR #97 analysis):

1. **Find All Instances First**:
   - [ ] Use grep/search to find ALL occurrences of the pattern before fixing
   - [ ] Document the locations found (file:line)
   - [ ] Example: `t_stat = effect / se if se > 0 else 0.0` appeared in 7 locations

2. **Fix Comprehensively in One Round**:
   - [ ] Fix ALL instances in the same PR/commit
   - [ ] Create a test that covers all locations
   - [ ] Don't fix incrementally across multiple review rounds

3. **Regression Test the Fix**:
   - [ ] Verify fix doesn't break other code paths
   - [ ] For early-return fixes: ensure downstream code still runs when needed
   - [ ] Example: Bootstrap early return must still compute per-effect SEs

4. **Common Patterns to Watch For**:
   - `if se > 0 else 0.0` -> should be `else np.nan` for undefined statistics
   - `if len(data) > 0 else return` -> check what downstream code expects
   - `mask = (condition)` -> verify mask logic for all parameter combinations

## Pre-Merge Review Checklist

Final checklist before approving a PR:

1. **Behavioral Completeness**:
   - [ ] Happy path tested
   - [ ] Edge cases tested (empty data, NaN inputs, boundary conditions)
   - [ ] Error/warning paths tested with behavioral assertions

2. **Inference Field Consistency**:
   - [ ] If one inference field (SE, t-stat, p-value) can be NaN, all related fields handle it
   - [ ] Aggregation methods propagate NaN correctly
   - [ ] Bootstrap methods handle NaN in base estimates

3. **Documentation Sync**:
   - [ ] Docstrings updated for all changed signatures
   - [ ] README updated if user-facing behavior changes
   - [ ] REGISTRY.md updated if methodology edge cases change

## Quick Reference: Common Patterns to Check

Before submitting methodology changes, verify these patterns:

```bash
# Find potential NaN handling issues (should use np.nan, not 0.0)
grep -n "if.*se.*>.*0.*else 0" diff_diff/*.py

# Find all t_stat calculations to ensure consistency
grep -n "t_stat.*=" diff_diff/*.py

# Find all inference field assignments
grep -n "\(se\|t_stat\|p_value\|ci_lower\|ci_upper\).*=" diff_diff/*.py | head -30
```
