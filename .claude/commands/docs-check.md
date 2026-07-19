---
description: Verify documentation completeness including scholarly references
argument-hint: "[all | readme | refs | api | tutorials | map]"
---

# Documentation Completeness Check

Verify that documentation is complete and includes appropriate scholarly references.

## Arguments

The user may provide an optional argument: `$ARGUMENTS`

- If empty or "all": Run all checks (including map validation)
- If "readme": Check the README catalog one-liner only
- If "refs" or "references": Check scholarly references in `docs/references.rst` only
- If "api": Check API documentation (RST files) only
- If "tutorials": Check tutorial coverage only
- If "map": Validate `docs/doc-deps.yaml` integrity only

## Documentation surface map (post 2026-04 docs refresh)

The README is a **landing page**, not the documentation. Each estimator/feature has documentation across multiple authoritative surfaces:

- **`diff_diff/guides/llms.txt`** - AI-agent contract (one-line catalog entry per estimator with paper citation + RTD link). Source of truth that mirrors into RTD via `html_extra_path` and into the wheel via `get_llm_guide()`.
- **`docs/api/*.rst`** - Sphinx API reference (autoclass).
- **`docs/references.rst`** - Bibliography (one entry per scholarly source, organized by sub-section).
- **`docs/tutorials/*.ipynb`** - Hands-on examples.
- **`README.md`** - **One-line catalog entry only** under `## Estimators` (or `## Diagnostics & Sensitivity` for diagnostic-class features). No usage examples, no parameter tables, no per-estimator section.

## Estimators and Required Documentation

The following estimators/features MUST have documentation:

### Core Estimators (require llms.txt entry + README catalog line + API docs + references)

| Estimator | llms.txt entry | README catalog | API RST | Reference Category |
|-----------|---------------|----------------|---------|-------------------|
| DifferenceInDifferences | "DifferenceInDifferences" | "DifferenceInDifferences" | estimators.rst | "Difference-in-Differences" |
| TwoWayFixedEffects | "TwoWayFixedEffects" | "TwoWayFixedEffects" | estimators.rst | "Two-Way Fixed Effects" |
| MultiPeriodDiD | "MultiPeriodDiD" | "MultiPeriodDiD" | estimators.rst | "Multi-Period and Staggered" |
| SyntheticDiD | "SyntheticDiD" | "SyntheticDiD" | estimators.rst | "Synthetic Difference-in-Differences" |
| CallawaySantAnna | "CallawaySantAnna" | "CallawaySantAnna" | staggered.rst | "Multi-Period and Staggered" |
| SunAbraham | "SunAbraham" | "SunAbraham" | staggered.rst | "Multi-Period and Staggered" |
| ImputationDiD | "ImputationDiD" | "ImputationDiD" | imputation.rst | "Multi-Period and Staggered" |
| TwoStageDiD | "TwoStageDiD" | "TwoStageDiD" | two_stage.rst | "Multi-Period and Staggered" |
| ChaisemartinDHaultfoeuille | "ChaisemartinDHaultfoeuille" | "ChaisemartinDHaultfoeuille" | chaisemartin_dhaultfoeuille.rst | "Multi-Period and Staggered" |
| EfficientDiD | "EfficientDiD" | "EfficientDiD" | efficient_did.rst | "Multi-Period and Staggered" |
| StackedDiD | "StackedDiD" | "StackedDiD" | stacked_did.rst | "Multi-Period and Staggered" |
| ContinuousDiD | "ContinuousDiD" | "ContinuousDiD" | continuous_did.rst | "Multi-Period and Staggered" |
| HeterogeneousAdoptionDiD | "HeterogeneousAdoptionDiD" | "HeterogeneousAdoptionDiD" | had.rst | "Heterogeneous Adoption (No-Untreated Designs)" |
| TripleDifference | "TripleDifference" | "TripleDifference" | triple_diff.rst | "Triple Difference" |
| StaggeredTripleDifference | "StaggeredTripleDifference" | "StaggeredTripleDifference" | staggered.rst | "Triple Difference" |
| WooldridgeDiD | "WooldridgeDiD" | "WooldridgeDiD" | wooldridge_etwfe.rst | "Multi-Period and Staggered" |
| TROP | "TROP" | "TROP" | trop.rst | "Triply Robust Panel" |
| HonestDiD | n/a (in `## Diagnostics`) | n/a (in `## Diagnostics`) | honest_did.rst | "Honest DiD" |
| BaconDecomposition | "BaconDecomposition" | "BaconDecomposition" | bacon.rst | "Multi-Period and Staggered" |

### Supporting Features (require llms.txt mention + API docs; README mention only if landing-page-relevant)

| Feature | llms.txt Mention | API RST |
|---------|------------------|---------|
| Wild bootstrap | "wild" and "bootstrap" | utils.rst |
| Cluster-robust SE | "cluster" | utils.rst |
| Parallel trends | "parallel trends" | utils.rst |
| Placebo tests | "placebo" | diagnostics.rst |
| Power analysis | "power" | power.rst |
| Pre-trends power | "pre-trends" or "pretrends" | pretrends.rst |

## Required Scholarly References

Each estimator category MUST have at least one scholarly reference in `docs/references.rst`:

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

### 2. llms.txt + README Catalog Check

For each estimator/diagnostic in the table above:

1. Read `diff_diff/guides/llms.txt` and verify the name appears under the right section:
   - **Estimators** (e.g. CallawaySantAnna, SunAbraham, TROP, BaconDecomposition): under `## Estimators`
   - **Diagnostics-class** (HonestDiD, and any future diagnostic-only entries): under `## Diagnostics and Sensitivity Analysis`
2. Read `README.md` and verify the name appears in the matching flat catalog:
   - **Estimators**: in the `## Estimators` section
   - **Diagnostics-class** (HonestDiD): in the `## Diagnostics & Sensitivity` section
3. Report missing entries

```bash
# Extract the README ## Estimators section. Use a flag-based awk because the
# range form `awk '/^## Estimators/,/^## /'` self-terminates on the opening H2.
extract_section() {
  awk -v target="$1" '
    $0 == "## " target { flag=1; next }
    flag && /^## / { flag=0 }
    flag { print }
  ' README.md
}

# Example: an estimator (lives in ## Estimators)
extract_section "Estimators" | grep -c 'CallawaySantAnna'

# Example: a diagnostic (lives in ## Diagnostics & Sensitivity)
extract_section "Diagnostics & Sensitivity" | grep -c 'Honest DiD'

# Always verify both surfaces
grep -c 'CallawaySantAnna' diff_diff/guides/llms.txt
```

Do NOT search for per-estimator README sections - they were intentionally removed in the 2026-04 docs refresh. The README's `## Estimators` and `## Diagnostics & Sensitivity` headings are the only valid catalog surfaces.

### 3. Scholarly References Check

For each reference category:
1. Search `docs/references.rst` for required citations (NOT README.md - the bibliography moved out of README in the 2026-04 docs refresh)
2. Verify author names and year appear together
3. Report missing references

Check patterns (case-insensitive, run against `docs/references.rst`):
- "Arkhangelsky.*2021" for Synthetic DiD
- "Callaway.*Sant.Anna.*2021" for staggered
- "Rambachan.*Roth.*2023" for Honest DiD
- "Athey.*Imbens.*Qu.*Viviano.*2025" for TROP
- "Goodman.Bacon.*2021" for Bacon decomposition
- etc.

```bash
# Example
grep -i 'Arkhangelsky.*2021' docs/references.rst
```

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

### 5b. Docs IA Registration Check (post-#703 information architecture)

The site navigation contract is CI-enforced by `tests/test_docs_ia.py` (docs-tests
workflow). Run it directly for immediate feedback:

```bash
PYTHONPATH=. DIFF_DIFF_BACKEND=python pytest tests/test_docs_ia.py -q
```

What it pins (fix at the source listed, never by widening the test):
1. `docs/index.rst` has ONE toctree = the 5 section landing pages. New top-level
   pages join a section landing page (`getting_started` / `practitioners` /
   `tutorials/index` / `user_guide` / `api/index`), never the root.
2. Every `docs/tutorials/*.ipynb` is registered in `docs/tutorials/index.rst`
   twice: a short-labeled toctree entry (`Label <stem>`, <= 40 chars) in its
   group AND a `grid-item-card` with `:link: <stem>`.
3. Homepage "Supported Estimators" table rows match the `api/index.rst`
   Estimators autosummary 1:1 - new estimators need BOTH.
4. Any `autoclass` documented with `:no-index:` on a module page needs a
   canonical autosummary entry in `api/index.rst`, else `:class:` xrefs to it
   render as dead text.

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

llms.txt + README Catalog:
  [PASS] DifferenceInDifferences - Found in llms.txt and README Estimators catalog
  [PASS] CallawaySantAnna - Found in both surfaces
  [FAIL] NewEstimator - missing from llms.txt and README catalog

Scholarly References (docs/references.rst):
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

### 8. Dependency Map Validation (if "map" or "all")

Validate the integrity of `docs/doc-deps.yaml`:

1. **Read and parse** `docs/doc-deps.yaml`. If missing or malformed YAML, report error.

2. **Check all doc paths exist**: For every `path` in every `sources` entry, verify the file
   exists on disk. Report missing files:
   ```
   [FAIL] docs/doc-deps.yaml references non-existent: docs/old_name.rst
   ```

3. **Check all source files have entries**: List all `diff_diff/*.py` and
   `diff_diff/visualization/*.py` files. For each, verify it appears either as a key in
   `sources:` or as a member of a `groups:` entry. Report missing:
   ```
   [WARN] diff_diff/new_module.py has no entry in docs/doc-deps.yaml
   ```

4. **Check for orphan doc paths**: Collect all unique doc paths from the map. Check if any
   doc file referenced in the map no longer exists or has been renamed.

5. **Report summary**:
   ```
   Dependency Map (docs/doc-deps.yaml):
     Sources mapped: 28
     Groups defined: 9
     Doc paths referenced: 45
     [PASS/FAIL] All doc paths exist
     [PASS/WARN] All source files have entries
   ```

## Notes

- This check is especially important after adding new estimators
- The CONTRIBUTING.md file documents what documentation is required for new features
- Missing references should cite the original methodology paper, not textbooks
- When adding new estimators, update this skill's tables and docs/doc-deps.yaml accordingly
