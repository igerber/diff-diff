---
description: Validate that tutorial Jupyter notebooks execute without errors
argument-hint: "[all | 01-10 | notebook-name]"
---

# Validate Tutorial Notebooks

Validate that tutorial Jupyter notebooks execute without errors.

## Arguments

The user may provide an optional argument: `$ARGUMENTS`

- If empty or not provided: Ask the user which notebook(s) to validate
- If "all": Validate all notebooks in `docs/tutorials/`
- If a number (e.g., "01", "1", "10"): Validate that specific notebook
- If a name (e.g., "basic_did", "trop"): Validate the matching notebook

## Available Notebooks

```
01_basic_did.ipynb        - Basic 2x2 DiD
02_staggered_did.ipynb    - Callaway-Sant'Anna staggered adoption
03_synthetic_did.ipynb    - Synthetic DiD
04_parallel_trends.ipynb  - Parallel trends testing
05_honest_did.ipynb       - Honest DiD sensitivity analysis
06_power_analysis.ipynb   - Power analysis for study design
07_pretrends_power.ipynb  - Pre-trends power analysis
08_triple_diff.ipynb      - Triple Difference (DDD)
09_real_world_examples.ipynb - Real-world datasets
10_trop.ipynb             - TROP estimator
```

## Instructions

1. **Parse the argument** to determine which notebook(s) to run:
   - If no argument provided, use AskUserQuestion to let user select:
     - Option 1: "All notebooks"
     - Option 2: "Select specific notebook" (then show numbered list)
   - If "all", run all notebooks
   - If a number, find the matching notebook (e.g., "01" or "1" matches "01_basic_did.ipynb")
   - If a partial name, find the matching notebook (e.g., "trop" matches "10_trop.ipynb")

2. **Execute each selected notebook** using:
   ```bash
   jupyter nbconvert --to notebook --execute --inplace "docs/tutorials/NOTEBOOK.ipynb" 2>&1
   ```

   Or if nbconvert is not available, try:
   ```bash
   python -m jupyter nbconvert --to notebook --execute --inplace "docs/tutorials/NOTEBOOK.ipynb" 2>&1
   ```

3. **Report results** for each notebook:
   - Success: "01_basic_did.ipynb - PASSED"
   - Failure: Show the error message and which cell failed

4. **Clear outputs and metadata** after validation completes (regardless of pass/fail):
   ```bash
   jupyter nbconvert --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True --inplace "docs/tutorials/NOTEBOOK.ipynb"
   ```

   This ensures notebook outputs and execution metadata (timestamps, execution counts) are not committed to git.

5. **Summary**: After all notebooks complete, show a summary like:
   ```
   Validation complete: 8/10 passed, 2 failed
   Failed: 05_honest_did.ipynb, 09_real_world_examples.ipynb

   All notebook outputs have been cleared.
   ```

## Notes

- Notebooks are executed to validate they work, then outputs and metadata are cleared
- If a notebook fails, continue to the next one (don't stop early)
- The working directory should be the project root
- Some notebooks may take a while to run (especially those with bootstrap)
