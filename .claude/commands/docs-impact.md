---
description: Show which documentation files may need updating based on changed source files
argument-hint: "[file1.py file2.py ...]"
---

# Documentation Impact Report

Identify which documentation files may need updating when source files in `diff_diff/` change.
Uses the dependency map at `docs/doc-deps.yaml`.

## Arguments

`$ARGUMENTS` is an optional space-separated list of source file paths (e.g., `diff_diff/staggered.py`).
If empty, auto-detect changed files from git.

## Instructions

### 1. Load Dependency Map

Read `docs/doc-deps.yaml` using the Read tool.

If the file does not exist or cannot be parsed, display:
```
Error: docs/doc-deps.yaml not found or malformed. Cannot generate impact report.
```
And stop.

### 2. Identify Changed Source Files

**If `$ARGUMENTS` is non-empty**: Use those file paths directly as the changed files list.

**If `$ARGUMENTS` is empty**: Auto-detect from git:

```bash
# Unstaged changes
git diff --name-only HEAD 2>/dev/null || true
# Staged changes
git diff --cached --name-only 2>/dev/null || true
# Untracked files
git ls-files --others --exclude-standard 2>/dev/null || true
```

Filter results to only files matching `diff_diff/**/*.py`. Deduplicate.

If no source files found, display:
```
No changed source files in diff_diff/ detected. Nothing to report.
```
And stop.

### 3. Resolve Group Membership

For each changed file, check if it appears in any `groups:` list in the YAML.
If it does, resolve it to the **first entry** in that group (the primary module).
This is the key used for doc lookup in the `sources:` section.

Example: if `diff_diff/staggered_bootstrap.py` changed, it resolves to `diff_diff/staggered.py`
because it is in the `staggered` group.

### 4. Look Up Impacted Docs

For each resolved source entry in the `sources:` section:
1. Get the `drift_risk` level
2. Get the list of `docs` entries (path, type, section, note)

Collect all impacted docs across all changed files. Deduplicate by path, but merge
section hints from different sources (e.g., REGISTRY.md may be referenced by both
staggered.py and survey.py with different section hints).

### 5. Validate Doc Paths

For each unique doc `path` in the results, verify the file exists on disk using the
Read tool (or Glob). If a path does not exist, flag it:
```
[STALE] docs/doc-deps.yaml references non-existent file: <path>
```

### 6. Display Report

Display results in priority order:

1. **METHODOLOGY (always warn)**: All docs with `type: methodology`, regardless of `drift_risk`.
   These are shown first because undocumented methodology deviations are P1 in AI review.
2. **HIGH DRIFT RISK**: Remaining docs (non-methodology) with `drift_risk: high`.
3. **MEDIUM DRIFT RISK**: Docs with `drift_risk: medium` (excluding methodology, already shown).
4. **LOW DRIFT RISK**: Docs with `drift_risk: low` (excluding methodology, already shown).

Within each group, show the type label and path, with section hints where available.

**Output format:**

```
=== Documentation Impact Report ===
Changed: <comma-separated list of changed source files>

METHODOLOGY (always warn):
  docs/methodology/REGISTRY.md -- <section hints>

HIGH DRIFT RISK:
  [roadmap] docs/survey-roadmap.md

MEDIUM DRIFT RISK:
  [user_guide] README.md -- <section hints>
  [tutorial] docs/tutorials/02_staggered_did.ipynb
  [performance] docs/benchmarks.rst

LOW DRIFT RISK:
  [api_reference] docs/api/staggered.rst

No map entry: <files not found in sources or groups, or "(none)">
Stale references: <invalid paths, or "(none)">
Always check: changelog.d/ fragment (never CHANGELOG.md's Unreleased directly), ROADMAP.md
```

### 7. Flag Missing Entries

List any changed source files that had no entry in the `sources:` section and were not
members of any group:
```
No map entry: diff_diff/new_module.py (consider adding to docs/doc-deps.yaml)
```

## Examples

```bash
# Auto-detect from git status
/docs-impact

# Check specific files
/docs-impact diff_diff/staggered.py diff_diff/survey.py
```
