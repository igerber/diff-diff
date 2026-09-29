# DiD OVB Sensitivity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a clean-room canonical DiD OVB sensitivity estimator in Python and verify its applied results against an independent R implementation.

**Architecture:** Add a focused estimator and results module that consumes a balanced two-period panel, uses existing cross-fitting/learner helpers, and exposes bias bounds plus RV/XRV and contour data. Keep the current `DMLDiD` contract unchanged; parity is a separate R script and committed fixture.

**Tech Stack:** Python, NumPy, pandas, SciPy, pytest, existing `diff_diff._crossfit` learners, Rscript for parity.

---

### Task 1: Add the failing public API contract tests

**Files:**
- Create: `tests/test_did_ovb_sensitivity.py`

- [ ] **Step 1: Write tests for two-period input normalization and missing-input errors.**

```python
def test_fit_accepts_balanced_two_period_panel_and_reports_short_att():
    result = DIDOVBSensitivity(n_folds=2, seed=7).fit(
        panel, outcome="y", treatment="treated", time="period",
        unit="unit", covariates=["x"],
    )
    assert np.isfinite(result.short_att)
    assert result.n_obs == len(panel) // 2

def test_fit_rejects_more_than_two_periods():
    with pytest.raises(ValueError, match="exactly two periods"):
        DIDOVBSensitivity().fit(panel_3_periods, outcome="y", treatment="treated",
                                time="period", unit="unit", covariates=["x"])
```

- [ ] **Step 2: Run the focused tests and confirm they fail because the API is absent.**

Run: `pytest -q tests/test_did_ovb_sensitivity.py`

Expected: collection or import failure naming `DIDOVBSensitivity`.

### Task 2: Implement input normalization and cross-fitted estimable components

**Files:**
- Create: `diff_diff/did_ovb.py`
- Modify: `diff_diff/__init__.py`
- Test: `tests/test_did_ovb_sensitivity.py`

- [ ] **Step 1: Add the smallest estimator skeleton and normalize a balanced panel.**

Implement `fit()` with explicit validation for unit uniqueness, two periods,
finite outcome/covariates, binary treatment, and at least one treated and one
control unit. Construct `delta_y` as post minus pre and retain one row per unit.

- [ ] **Step 2: Add failing tests for deterministic cross-fitting and component identities.**

```python
def test_components_obey_scale_identity():
    result = DIDOVBSensitivity(n_folds=2, seed=11).fit(...)
    assert result.scale == pytest.approx(
        np.sqrt(result.sigma2_control * result.nu2_selection)
    )
    assert result.short_att == pytest.approx(result.short_att_against_control)
```

- [ ] **Step 3: Implement cross-fitted nuisance predictions and orthogonal plug-ins.**

Use `assign_folds` and `cross_fit_predict` with the existing learner factory;
estimate the control trend regression and treatment propensity, compute the
paper's short ATT, residual control variance, selection scale, and retained
diagnostics. Keep all arrays in the result only when needed for delta-method
variance and parity diagnostics.

- [ ] **Step 4: Run focused tests and make them pass.**

Run: `pytest -q tests/test_did_ovb_sensitivity.py -k 'normaliz or component'`

### Task 3: Add bounds, inference, RV/XRV, and structured result output

**Files:**
- Modify: `diff_diff/did_ovb.py`
- Modify: `diff_diff/__init__.py`
- Test: `tests/test_did_ovb_sensitivity.py`

- [ ] **Step 1: Add failing tests for bounds and monotone robustness searches.**

```python
def test_bounds_are_symmetric_and_expand_with_strength():
    result = DIDOVBSensitivity(n_folds=2, seed=7).fit(...)
    narrow = result.bounds(trend_strength=.2, selection_strength=.2)
    wide = result.bounds(trend_strength=.5, selection_strength=.5)
    assert narrow.lower < result.short_att < narrow.upper
    assert wide.radius > narrow.radius

def test_null_inside_original_interval_has_zero_rv_and_xrv():
    result = DIDOVBSensitivity(n_folds=2, seed=7).fit(...)
    out = result.robustness_value(null_value=result.short_att, alpha=.05)
    assert out.rv == 0.0
    assert out.xrv == 0.0
```

- [ ] **Step 2: Implement result dataclasses and normal/delta-method intervals.**

Expose `bounds()`, `robustness_value()`, `contour()`, `summary()`,
`to_dict()`, and `to_dataframe()`. Validate strengths in `[0, 1]`, alpha in
`(0, .5)`, and return explicit NaNs/errors for non-identifiable fits.

- [ ] **Step 3: Implement RV/XRV grid search against the paper's widest-CI definition.**

Use a deterministic one-dimensional search over common strength for RV and
selection-only strength for XRV, retaining the grid and CI endpoints used to
make the decision. Do not silently replace a failed endpoint with a point
estimate.

- [ ] **Step 4: Run focused tests and then the full relevant Python suite.**

Run: `pytest -q tests/test_did_ovb_sensitivity.py tests/test_dml_did.py tests/test_methodology_dml_did.py`

### Task 4: Add R application parity harness

**Files:**
- Create: `benchmarks/R/generate_did_ovb_parity.R`
- Create: `benchmarks/data/did_ovb_parity_panel.csv`
- Create: `benchmarks/data/did_ovb_r_results.json`
- Create: `tests/test_did_ovb_r_parity.py`
- Modify: `benchmarks/R/requirements.R`
- Modify: `benchmarks/README.md`

- [ ] **Step 1: Write the parity test before the R oracle exists.**

The test must load the committed panel and R JSON, run the Python estimator with
the exact recorded settings, and compare short ATT, sigma/nu, scale, bounds,
RV, and XRV with named tolerances. Skip only when `Rscript` or the required R
package is unavailable; fail if the fixture is missing or malformed.

- [ ] **Step 2: Implement the R script from public formulas and fixed settings.**

The script writes JSON containing settings, sample counts, intermediate
components, and final sensitivity outputs. It must not import Python or call
the GPL-3 source through an undocumented bridge.

- [ ] **Step 3: Generate the fixture and run Python/R parity.**

Run: `Rscript benchmarks/R/generate_did_ovb_parity.R`

Then: `pytest -q tests/test_did_ovb_r_parity.py`

Expected: parity passes with all component-level comparisons inside the declared tolerance.

### Task 5: Document, audit, and prepare the branch

**Files:**
- Modify: `docs/api/dml_did.rst` or create `docs/api/did_ovb.rst`
- Modify: `docs/methodology/REGISTRY.md`
- Modify: `README.md` only if the public API is promoted there
- Modify: `LICENSE` notices only if reused MIT material is included

- [ ] **Step 1: Document clean-room provenance, citations, limitations, and R parity command.**
- [ ] **Step 2: Run formatting/type checks and the complete targeted test set.**
- [ ] **Step 3: Inspect `git diff`, verify no GPL-3 source was copied, and confirm the working tree contains only intended files.**
- [ ] **Step 4: Commit locally with a focused message.**
- [ ] **Step 5: Do not push until the user explicitly asks for the reviewed commit/PR.**
