---
description: Verify documentation completeness including scholarly references
argument-hint: "[all | readme | refs | api | tutorials]"
---

# Documentation Completeness Check

Verify that documentation is complete and includes appropriate scholarly references.

## Arguments

The user may provide an optional argument: `$ARGUMENTS`

- If empty or "all": Run all checks
- If "readme": Check README.md sections only
- If "refs" or "references": Check scholarly references only
- If "api": Check API documentation (RST files) only
- If "tutorials": Check tutorial coverage only

## Estimators and Required Documentation

The following estimators/features MUST have documentation:

### Core Estimators (require README section + API docs + references)

| Estimator | README Section | API RST | Reference Category |
|-----------|----------------|---------|-------------------|
| DifferenceInDifferences | "Basic Difference-in-Differences" | estimators.rst | "Difference-in-Differences" |
| TwoWayFixedEffects | "Two-Way Fixed Effects" | estimators.rst | "Two-Way Fixed Effects" |
| MultiPeriodDiD | "Multi-Period" | estimators.rst | "Multi-Period and Staggered" |
| SyntheticDiD | "Synthetic DiD" or "Synthetic Difference" | estimators.rst | "Synthetic Difference-in-Differences" |
| CallawaySantAnna | "Callaway" or "Staggered" | staggered.rst | "Multi-Period and Staggered" |
| SunAbraham | "Sun" and "Abraham" | staggered.rst | "Multi-Period and Staggered" |
| TripleDifference | "Triple Diff" or "DDD" | triple_diff.rst | "Triple Difference" |
| TROP | "TROP" or "Triply Robust" | trop.rst | "Triply Robust Panel" |
| HonestDiD | "Honest DiD" or "sensitivity" | honest_did.rst | "Honest DiD" |
| BaconDecomposition | "Bacon" or "decomposition" | estimators.rst | "Multi-Period and Staggered" |

### Supporting Features (require README mention + API docs)

| Feature | README Mention | API RST |
|---------|----------------|---------|
| Wild bootstrap | "wild" and "bootstrap" | utils.rst |
| Cluster-robust SE | "cluster" | utils.rst |
| Parallel trends | "parallel trends" | utils.rst |
| Placebo tests | "placebo" | diagnostics.rst |
| Power analysis | "power" | power.rst |
| Pre-trends power | "pre-trends" or "pretrends" | pretrends.rst |

## Required Scholarly References

Each estimator category MUST have at least one scholarly reference in README.md:

### Reference Requirements

```
Difference-in-Differences:
  - Card & Krueger (1994) OR Ashenfelter & Card (1985)

Two-Way Fixed Effects:
  - Wooldridge (2010) OR Imai & Kim (2021)

Synthetic Difference-in-Differences:
  - Arkhangelsky et al. (2021)

Callaway-Sant'Anna / Staggered:
  - Callaway & Sant'Anna (2021)

Sun-Abraham:
  - Sun & Abraham (2021)

Triple Difference:
  - Ortiz-Villavicencio & Sant'Anna (2025) OR Olden & Møen (2022)

TROP:
  - Athey, Imbens, Qu & Viviano (2025)

Honest DiD:
  - Rambachan & Roth (2023)

Pre-trends Power:
  - Roth (2022)

Wild Bootstrap:
  - Cameron, Gelbach & Miller (2008) OR Webb (2014)

Goodman-Bacon Decomposition:
  - Goodman-Bacon (2021)
```

## Instructions

### 1. Parse Arguments

Determine which checks to run based on `$ARGUMENTS`.

### 2. README Section Check

For each estimator in the table above:
1. Read README.md
2. Search for the required section/mention (case-insensitive)
3. Report missing sections

```bash
# Example: Check if "Callaway" appears in README
grep -i "callaway" README.md
```

### 3. Scholarly References Check

For each reference category:
1. Search README.md References section for required citations
2. Verify author names and year appear together
3. Report missing references

Check patterns (case-insensitive):
- "Arkhangelsky.*2021" for Synthetic DiD
- "Callaway.*Sant.Anna.*2021" for staggered
- "Rambachan.*Roth.*2023" for Honest DiD
- "Athey.*Imbens.*Qu.*Viviano.*2025" for TROP
- "Goodman.Bacon.*2021" for Bacon decomposition
- etc.

### 4. API Documentation Check

For each RST file in `docs/api/`:
1. Verify the file exists
2. Check it contains `autoclass` or `autofunction` directives
3. Report missing or empty API docs

```bash
# List API docs
ls docs/api/*.rst

# Check for autoclass directives
grep -l "autoclass" docs/api/*.rst
```

### 5. Tutorial Coverage Check

For each major feature, verify a tutorial covers it:

| Feature | Tutorial |
|---------|----------|
| Basic DiD | 01_basic_did.ipynb |
| Staggered | 02_staggered_did.ipynb |
| Synthetic DiD | 03_synthetic_did.ipynb |
| Parallel trends | 04_parallel_trends.ipynb |
| Honest DiD | 05_honest_did.ipynb |
| Power analysis | 06_power_analysis.ipynb |
| Pre-trends | 07_pretrends_power.ipynb |
| Triple Diff | 08_triple_diff.ipynb |
| TROP | 10_trop.ipynb |

Check each tutorial file exists and is non-empty.

### 6. Cross-Reference Check

For estimators added to the codebase, verify they have:
1. A class in `diff_diff/*.py`
2. Tests in `tests/test_*.py`
3. README documentation
4. API RST documentation
5. Scholarly reference (if method-based)

To find all public estimator classes:
```bash
grep -r "^class.*Estimator\|^class.*DiD\|^class.*Results" diff_diff/*.py
```

### 7. Report Results

Generate a summary report:

```
=== Documentation Completeness Check ===

README Sections:
  [PASS] DifferenceInDifferences - Found in "Basic Difference-in-Differences"
  [PASS] CallawaySantAnna - Found in "Staggered Adoption"
  [FAIL] NewEstimator - NOT FOUND

Scholarly References:
  [PASS] Synthetic DiD - Arkhangelsky et al. (2021)
  [PASS] Honest DiD - Rambachan & Roth (2023)
  [FAIL] Bacon Decomposition - Missing Goodman-Bacon (2021)

API Documentation:
  [PASS] docs/api/estimators.rst - Contains autoclass directives
  [PASS] docs/api/staggered.rst - Contains autoclass directives
  [FAIL] docs/api/new_module.rst - File missing

Tutorial Coverage:
  [PASS] Basic DiD - 01_basic_did.ipynb exists
  [PASS] TROP - 10_trop.ipynb exists

Summary: 15/18 checks passed, 3 issues found
```

## Notes

- This check is especially important after adding new estimators
- The CLAUDE.md file documents what documentation is required for new features
- Missing references should cite the original methodology paper, not textbooks
- When adding new estimators, update this skill's tables accordingly
