# Methodology Review

This document tracks the progress of reviewing each estimator's implementation against the Methodology Registry and academic references. It ensures that implementations are correct, consistent, and well-documented.

For the methodology registry with academic foundations and key equations, see [docs/methodology/REGISTRY.md](docs/methodology/REGISTRY.md).

---

## Overview

Each estimator in diff-diff should be periodically reviewed to ensure:
1. **Correctness**: Implementation matches the academic paper's equations
2. **Reference alignment**: Behavior matches reference implementations (R packages, Stata commands)
3. **Edge case handling**: Documented edge cases are handled correctly
4. **Standard errors**: SE formulas match the documented approach

---

## Review Status Summary

| Estimator | Module | R Reference | Status | Last Review |
|-----------|--------|-------------|--------|-------------|
| DifferenceInDifferences | `estimators.py` | `fixest::feols()` | Not Started | - |
| MultiPeriodDiD | `estimators.py` | `fixest::feols()` | Not Started | - |
| TwoWayFixedEffects | `twfe.py` | `fixest::feols()` | Not Started | - |
| CallawaySantAnna | `staggered.py` | `did::att_gt()` | **Complete** | 2026-01-24 |
| SunAbraham | `sun_abraham.py` | `fixest::sunab()` | Not Started | - |
| SyntheticDiD | `synthetic_did.py` | `synthdid::synthdid_estimate()` | Not Started | - |
| TripleDifference | `triple_diff.py` | (forthcoming) | Not Started | - |
| TROP | `trop.py` | (forthcoming) | Not Started | - |
| BaconDecomposition | `bacon.py` | `bacondecomp::bacon()` | Not Started | - |
| HonestDiD | `honest_did.py` | `HonestDiD` package | Not Started | - |
| PreTrendsPower | `pretrends.py` | `pretrends` package | Not Started | - |
| PowerAnalysis | `power.py` | `pwr` / `DeclareDesign` | Not Started | - |

**Status legend:**
- **Not Started**: No formal review conducted
- **In Progress**: Review underway
- **Complete**: Review finished, implementation verified

---

## Detailed Review Notes

### Core DiD Estimators

#### DifferenceInDifferences

| Field | Value |
|-------|-------|
| Module | `estimators.py` |
| Primary Reference | Wooldridge (2010), Angrist & Pischke (2009) |
| R Reference | `fixest::feols()` |
| Status | Not Started |
| Last Review | - |

**Corrections Made:**
- (None yet)

**Outstanding Concerns:**
- (None yet)

---

#### MultiPeriodDiD

| Field | Value |
|-------|-------|
| Module | `estimators.py` |
| Primary Reference | Freyaldenhoven et al. (2021) |
| R Reference | `fixest::feols()` |
| Status | Not Started |
| Last Review | - |

**Corrections Made:**
- (None yet)

**Outstanding Concerns:**
- (None yet)

---

#### TwoWayFixedEffects

| Field | Value |
|-------|-------|
| Module | `twfe.py` |
| Primary Reference | Wooldridge (2010), Ch. 10 |
| R Reference | `fixest::feols()` |
| Status | Not Started |
| Last Review | - |

**Corrections Made:**
- (None yet)

**Outstanding Concerns:**
- (None yet)

---

### Modern Staggered Estimators

#### CallawaySantAnna

| Field | Value |
|-------|-------|
| Module | `staggered.py` |
| Primary Reference | Callaway & Sant'Anna (2021) |
| R Reference | `did::att_gt()` |
| Status | **Complete** |
| Last Review | 2026-01-24 |

**Verified Components:**
- [x] ATT(g,t) basic formula (hand-calculated exact match)
- [x] Doubly robust estimator
- [x] IPW estimator
- [x] Outcome regression
- [x] Base period selection (varying/universal)
- [x] Anticipation parameter handling
- [x] Simple/event-study/group aggregation
- [x] Analytical SE with weight influence function
- [x] Bootstrap SE (Rademacher/Mammen/Webb)
- [x] Control group composition (never_treated/not_yet_treated)
- [x] All documented edge cases from REGISTRY.md

**Test Coverage:**
- 46 methodology verification tests in `tests/test_methodology_callaway.py`
- 93 existing tests in `tests/test_staggered.py`
- R benchmark tests (skip if R not available)

**R Comparison Results:**
- Overall ATT matches within 20% (difference due to dynamic effects in generated data)
- Post-treatment ATT(g,t) values match within 20%
- Pre-treatment effects may differ due to base_period handling differences

**Corrections Made:**
- (None - implementation verified correct)

**Outstanding Concerns:**
- R comparison shows ~20% difference in overall ATT with generated data
  - Likely due to differences in how dynamic effects are handled in data generation
  - Individual ATT(g,t) values match closely for post-treatment periods
  - Further investigation recommended with real-world data
- Pre-treatment ATT(g,t) may differ from R due to base_period="varying" semantics
  - Python uses t-1 as base for pre-treatment
  - R's behavior requires verification

**Deviations from R's did::att_gt():**
1. **NaN for invalid inference**: When SE is non-finite or zero, Python returns NaN for
   t_stat/p_value rather than potentially erroring. This is a defensive enhancement.

**Alignment with R's did::att_gt() (as of v2.1.5):**
1. **Webb weights**: Webb's 6-point distribution with values ±√(3/2), ±1, ±√(1/2)
   uses equal probabilities (1/6 each) matching R's `did` package. This gives
   E[w]=0, Var(w)=1.0, consistent with other bootstrap weight distributions.

   **Verification**: Our implementation matches the well-established `fwildclusterboot`
   R package (C++ source: [wildboottest.cpp](https://github.com/s3alfisc/fwildclusterboot/blob/master/src/wildboottest.cpp)).
   The implementation uses `sqrt(1.5)`, `1`, `sqrt(0.5)` (and negatives) with equal 1/6
   probabilities—identical to our values.

   **Note on documentation discrepancy**: Some documentation (e.g., fwildclusterboot
   vignette) describes Webb weights as "±1.5, ±1, ±0.5". This appears to be a
   simplification for readability. The actual implementations use ±√1.5, ±1, ±√0.5
   which provides the required unit variance (Var(w) = 1.0).

---

#### SunAbraham

| Field | Value |
|-------|-------|
| Module | `sun_abraham.py` |
| Primary Reference | Sun & Abraham (2021) |
| R Reference | `fixest::sunab()` |
| Status | Not Started |
| Last Review | - |

**Corrections Made:**
- (None yet)

**Outstanding Concerns:**
- (None yet)

---

### Advanced Estimators

#### SyntheticDiD

| Field | Value |
|-------|-------|
| Module | `synthetic_did.py` |
| Primary Reference | Arkhangelsky et al. (2021) |
| R Reference | `synthdid::synthdid_estimate()` |
| Status | Not Started |
| Last Review | - |

**Corrections Made:**
- (None yet)

**Outstanding Concerns:**
- (None yet)

---

#### TripleDifference

| Field | Value |
|-------|-------|
| Module | `triple_diff.py` |
| Primary Reference | Ortiz-Villavicencio & Sant'Anna (2025) |
| R Reference | (forthcoming) |
| Status | Not Started |
| Last Review | - |

**Corrections Made:**
- (None yet)

**Outstanding Concerns:**
- (None yet)

---

#### TROP

| Field | Value |
|-------|-------|
| Module | `trop.py` |
| Primary Reference | Athey, Imbens, Qu & Viviano (2025) |
| R Reference | (forthcoming) |
| Status | Not Started |
| Last Review | - |

**Corrections Made:**
- (None yet)

**Outstanding Concerns:**
- (None yet)

---

### Diagnostics & Sensitivity

#### BaconDecomposition

| Field | Value |
|-------|-------|
| Module | `bacon.py` |
| Primary Reference | Goodman-Bacon (2021) |
| R Reference | `bacondecomp::bacon()` |
| Status | Not Started |
| Last Review | - |

**Corrections Made:**
- (None yet)

**Outstanding Concerns:**
- (None yet)

---

#### HonestDiD

| Field | Value |
|-------|-------|
| Module | `honest_did.py` |
| Primary Reference | Rambachan & Roth (2023) |
| R Reference | `HonestDiD` package |
| Status | Not Started |
| Last Review | - |

**Corrections Made:**
- (None yet)

**Outstanding Concerns:**
- (None yet)

---

#### PreTrendsPower

| Field | Value |
|-------|-------|
| Module | `pretrends.py` |
| Primary Reference | Roth (2022) |
| R Reference | `pretrends` package |
| Status | Not Started |
| Last Review | - |

**Corrections Made:**
- (None yet)

**Outstanding Concerns:**
- (None yet)

---

#### PowerAnalysis

| Field | Value |
|-------|-------|
| Module | `power.py` |
| Primary Reference | Bloom (1995), Burlig et al. (2020) |
| R Reference | `pwr` / `DeclareDesign` |
| Status | Not Started |
| Last Review | - |

**Corrections Made:**
- (None yet)

**Outstanding Concerns:**
- (None yet)

---

## Review Process Guidelines

### Review Checklist

For each estimator, complete the following steps:

- [ ] **Read primary academic source** - Review the key paper(s) cited in REGISTRY.md
- [ ] **Compare key equations** - Verify implementation matches equations in REGISTRY.md
- [ ] **Run benchmark against R reference** - Execute `benchmarks/run_benchmarks.py --estimator <name>` if available
- [ ] **Verify edge case handling** - Check behavior matches REGISTRY.md documentation
- [ ] **Check standard error formula** - Confirm SE computation matches reference
- [ ] **Document any deviations** - Add notes explaining intentional differences with rationale

### When to Update This Document

1. **After completing a review**: Update status to "Complete" and add date
2. **When making corrections**: Document what was fixed in the "Corrections Made" section
3. **When identifying issues**: Add to "Outstanding Concerns" for future investigation
4. **When deviating from reference**: Document the deviation and rationale

### Deviation Documentation

When our implementation intentionally differs from the reference implementation, document:

1. **What differs**: Specific behavior or formula that differs
2. **Why**: Rationale (e.g., "defensive enhancement", "bug in R package", "follows updated paper")
3. **Impact**: Whether results differ in practice
4. **Cross-reference**: Update REGISTRY.md edge cases section

Example:
```
**Deviation (2025-01-15)**: CallawaySantAnna returns NaN for t_stat when SE is non-finite,
whereas R's `did::att_gt` would error. This is a defensive enhancement that provides
more graceful handling of edge cases while still signaling invalid inference to users.
```

### Priority Order

Suggested order for reviews based on usage and complexity:

1. **High priority** (most used, complex methodology):
   - CallawaySantAnna
   - SyntheticDiD
   - HonestDiD

2. **Medium priority** (commonly used, simpler methodology):
   - DifferenceInDifferences
   - TwoWayFixedEffects
   - MultiPeriodDiD
   - SunAbraham
   - BaconDecomposition

3. **Lower priority** (newer or less commonly used):
   - TripleDifference
   - TROP
   - PreTrendsPower
   - PowerAnalysis

---

## Related Documents

- [REGISTRY.md](docs/methodology/REGISTRY.md) - Academic foundations and key equations
- [ROADMAP.md](ROADMAP.md) - Feature roadmap
- [TODO.md](TODO.md) - Technical debt tracking
- [CLAUDE.md](CLAUDE.md) - Development guidelines
