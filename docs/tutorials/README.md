# diff-diff tutorials

Jupyter notebooks that teach the library hands-on. The rendered versions, with
their outputs, are on Read the Docs:
<https://diff-diff.readthedocs.io/en/stable/tutorials/index.html>.

- **What is here and in what order:** [`index.rst`](index.rst) lists every
  notebook with a one-line description, grouped into Business Applications,
  Fundamentals, Advanced Methods and Study Design. The number in a filename is
  a stable id, not a reading order.
- **Writing or editing a notebook:** follow "Tutorial conventions" in
  [`CONTRIBUTING.md`](../../CONTRIBUTING.md), which also covers how to
  re-execute a notebook and update its drift test. The mechanical rules are
  enforced by `tests/test_tutorial_conventions.py`.

## Running the notebooks

```bash
pip install diff-diff matplotlib jupyter
jupyter notebook
```

Every notebook is self-contained (data is simulated or loaded by the library)
and runs on Python 3.9+. Three need an extra package that is not a diff-diff
dependency: `26_composition_drift_calibration` (`balance`),
`29_mmm_calibration_pymc` (`pymc-marketing`) and `30_mmm_calibration_meridian`
(`google-meridian`); each notebook states its exact install line, and the two
MMM notebooks need Python 3.12 and separate environments.
