### Internal
- **Tutorial conventions and a CI guard for them**: `CONTRIBUTING.md` gains a
  "Tutorial conventions" section (what every notebook states and how: estimand
  and assumptions, the truth check on simulated data, clustered inference, one
  event-study idiom, warnings, committed outputs, a drift test per notebook).
  `tests/test_tutorial_conventions.py` enforces the mechanical rules over every
  notebook by parsing the notebook JSON, with a shrinking allowlist of the
  violations that exist today; it runs in the docs-tests workflow. The notebooks
  workflow now runs every tutorial drift test, so a notebook-only change can no
  longer leave its pinned numbers unchecked. `docs/tutorials/README.md` is
  reduced to a pointer to the tutorial index and the conventions.
