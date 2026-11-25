# Contributing to diff-diff

## Linting and Formatting

Run before pushing — the ungated `Lint` workflow enforces these on every PR
push, and its `Lint Gate` job is a required status check:

```bash
ruff check diff_diff tests
black diff_diff tests
mypy diff_diff
```

- Tool versions are **pinned exactly** in the `dev` extra of `pyproject.toml`
  (`black`, `ruff`, `mypy`) and mirrored in `.github/workflows/lint.yml`
  (all three; the sync is enforced by `TestLintWorkflowPinSync`). `mypy
  diff_diff` is enforced at zero errors by the Lint workflow's Mypy job.
  A version bump is a deliberate PR updating both surfaces together.
  Refresh with `pip install -e ".[dev]"`. The pinned tools require
  Python >= 3.10 (dev tooling only; the library itself still supports
  Python 3.9).
- `[tool.ruff.lint] select = ["E", "F", "I", "W"]` is deliberately explicit and
  is **load-bearing**, not a restatement of ruff's defaults. Ruff 0.16.0 expanded
  its default selection from 59 rules to 413 across 38 linter families; the
  explicit `select` is what keeps a ruff upgrade from silently enabling hundreds
  of new rules. Do not drop it to inherit the defaults. Widening the rule set is
  a deliberate PR of its own.
- `[tool.ruff.lint.per-file-ignores]` entries are deliberate (trop
  logger-before-imports E402, honest_did math-notation `l` E741, `__init__`
  re-export F401, conftest import ordering E402). Do not "fix" those patterns
  or remove the ignores.
- The 2026-07 repo-wide normalization commits are listed in
  `.git-blame-ignore-revs`; run
  `git config blame.ignoreRevsFile .git-blame-ignore-revs` once so
  `git blame` attributes lines to their real authors.
- Branches created before the normalization commits: rebase onto main, then
  run `ruff check --fix` + `black` on your touched files before pushing.

## Documentation Requirements

When implementing new functionality, **always include accompanying documentation updates**.

### README is a landing page, not the docs

`README.md` is the GitHub/PyPI first-impression surface. Keep it lean (~190 lines). Most new content does NOT belong here.

**Only edit `README.md` when**:
- A new estimator is added (one line in the `## Estimators` flat catalog)
- A new top-level capability lands (one paragraph in `## Diagnostics & Sensitivity` or `## Survey Support`)
- Hero image, badges, or top-of-fold value-prop changes
- Documentation links rot

If you find yourself adding a usage example, a parameter table, or a multi-paragraph explanation to the README, you are in the wrong file - those belong on RTD or in `diff_diff/guides/llms.txt`.

### For New Estimators or Major Features

1. **`diff_diff/guides/llms.txt`** (AI-agent source of truth) - Add:
   - One-line catalog entry in the `## Estimators` section with paper citation + RTD link
   - One-line entry in `## Diagnostics and Sensitivity Analysis` if applicable
   - This file is published on RTD via `docs/conf.py` `html_extra_path` and bundled in the wheel via `get_llm_guide()` - it is the canonical machine-readable contract

2. **`docs/api/*.rst`** (technical source of truth) - Add:
   - RST documentation with `autoclass` directives
   - Method summaries
   - References to academic papers

3. **`docs/tutorials/*.ipynb`** - Update relevant tutorial or create new one:
   - Working code examples
   - Explanation of when/why to use the feature
   - Follow the "Tutorial conventions" section below (CI-enforced by
     `tests/test_tutorial_conventions.py`), including a drift test for a new notebook
   - New notebooks are registered in `docs/tutorials/index.rst` (NOT `docs/index.rst`):
     add a toctree entry with a short display label to the matching group
     (Business Applications / Fundamentals / Advanced Methods / Study Design)
     plus a `grid-item-card` in that group's card grid

4. **`docs/references.rst`** (bibliography source of truth) - Add:
   - Full citation under the appropriate sub-section (matches the `### Subsection` headings already in that file)
   - Use the RST format: `**Author (Year).** "Title." *Journal*, vol(num), pages. <https://doi.org/X>`

5. **`README.md`** - Add ONLY:
   - One line in the `## Estimators` catalog with the paper citation and RTD link

6. **`changelog.d/`** - Add a release-note fragment `changelog.d/YYYYMMDD-<slug>.md`
   (see `changelog.d/README.md` for the format). NEVER edit `CHANGELOG.md`'s
   `## [Unreleased]` section directly - it is pointer-only and CI-enforced.

7. **`CLAUDE.md`** - Update only if adding new critical rules or design patterns.

8. **`ROADMAP.md`** - Update only if shipping moves an item from planned to current.

9. **`docs/doc-deps.yaml`** - Add source-to-doc mappings for the new module.

### Docs IA invariants (CI-enforced by `tests/test_docs_ia.py`)

The documentation site's information architecture is machine-enforced; a red
`docs-tests` check means one of these was violated:

1. The root `docs/index.rst` toctree lists ONLY the 5 section landing pages
   (Getting Started / Practitioner Guide / Tutorials / User Guide / API
   Reference). New top-level pages join a section landing page, never the root -
   every root entry becomes a navbar item and past 5 the theme regrows an
   unusable "More" dropdown.
2. Every tutorial notebook has BOTH a short-labeled toctree entry (<= 40 chars)
   and a `grid-item-card`, in the SAME group of `docs/tutorials/index.rst`.
3. The homepage "Supported Estimators" table stays 1:1 with the
   `docs/api/index.rst` Estimators autosummary - new estimators need both.
4. A class documented with `:no-index:` on a module page must keep a canonical
   autosummary entry in `docs/api/index.rst`, else `:class:` cross-references
   to it render as dead text.

### Changelog fragments (CI-enforced by `tests/test_changelog_fragments.py`)

Release notes are per-PR fragment files under `changelog.d/` so that parallel
PRs never conflict on `CHANGELOG.md`:

1. Write your entry as `changelog.d/YYYYMMDD-<slug>.md` - one file per PR,
   `### <Category>` blocks with the bullets exactly as they should appear in
   the changelog. Format spec: `changelog.d/README.md`.
2. `CHANGELOG.md`'s `## [Unreleased]` section stays pointer-only between
   releases; any direct edit fails the docs-tests lane.
3. At release, `/bump-version` runs
   `.claude/scripts/changelog_compile.py compile`, which merges the fragments
   into the new version section (oldest-first within each category) and
   deletes them.
4. Rebasing a branch created before this system: move your old
   `## [Unreleased]` bullet into a new fragment file.

Rollback (if the system is ever removed): paste the current fragments' blocks
back under `## [Unreleased]` (dropping the pointer comment), delete
`changelog.d/`, `.claude/scripts/changelog_compile.py`, and
`tests/test_changelog_fragments.py`, revert the `docs-tests.yml` /
`rust-test.yml` filter entries and step, and restore the pre-change
`bump-version` steps from git history.

### For Bug Fixes or Minor Enhancements

- Update relevant docstrings
- Add/update tests
- Add a `changelog.d/YYYYMMDD-<slug>.md` fragment (never edit `## [Unreleased]` directly)
- **If methodology-related**: Update `docs/methodology/REGISTRY.md` edge cases section
- **README is almost never the right place** - skip it unless the bug was in a README claim

### Scholarly References

For methods based on academic papers, always include:
- Full citation in **`docs/references.rst`** under the appropriate `### Subsection` heading (NOT in README)
- Reference in RST API docs with paper details
- Citation in tutorial summary
- Optional: methodology reference in `docs/methodology/REGISTRY.md` for non-trivial design choices

Example format (RST):
```
- **Sun, L., & Abraham, S. (2021).** "Estimating Dynamic Treatment Effects in Event Studies with Heterogeneous Treatment Effects." *Journal of Econometrics*, 225(2), 175-199. https://doi.org/10.1016/j.jeconom.2020.09.006
```

## Tutorial conventions

The notebooks in `docs/tutorials/` are teaching material: a reader should meet
the same idea the same way wherever it comes up. These conventions apply to
every new notebook and to any notebook you edit. The mechanical ones are
CI-enforced by `tests/test_tutorial_conventions.py`; that guard carries a
shrinking allowlist of pre-existing violations (`KNOWN_VIOLATIONS`) - when you
repair a notebook, delete its entries; never add one.

### Teaching

1. **Opening.** An H1 that matches the notebook's card label in
   `docs/tutorials/index.rst`; one paragraph stating the question the notebook
   answers; a short "What you'll learn" list; a prerequisites line linking the
   tutorials it builds on.
2. **Estimand and assumptions before the first fit.** Name the target
   parameter, state the identifying assumptions in plain words, and say what
   would violate them.
3. **Check against the truth on synthetic data.** State the true effect as a
   named constant where the data is built, and pass every generator argument
   that determines the truth explicitly, even at its default value
   (`generate_staggered_data` defaults to `dynamic_effects=True`, so the true
   average effect is not `treatment_effect`). Compare the estimate and its
   interval to the truth in the output; if the interval misses, say why. Real
   data comes from a `load_*` loader where one exists. No data files are added
   under `docs/tutorials/`.
4. **Inference.** Cluster panel fits at the unit. `DifferenceInDifferences`
   does not auto-cluster, so pass `cluster=`; the estimators that cluster at
   the unit by default need nothing extra. Tutorial 01 explains why; elsewhere
   one line is enough, and any other choice is justified in prose. Bootstraps
   use `n_bootstrap=999` and a fixed seed (fewer only for runtime, and say so).
   When prose discusses a p-value, name the inference that produced it and
   check it against the estimator's REGISTRY section, because the count-based
   p-values differ. With `B` valid draws of which `c` are at least as extreme
   as the estimate: the permutation test reports `(c + 1) / (B + 1)` (steps of
   `1/(B+1)`, smallest value `1/(B+1)`); the SyntheticDiD placebo reports
   `max(c / B, 1/(B+1))` (steps of `1/B`, same floor). A p-value computed
   analytically from a resampled standard error (SyntheticDiD bootstrap and
   jackknife, ChangesInChanges) is not count-based and has no floor.
5. **Pre-trend tests are not pass/fail.** A non-rejection is not evidence that
   parallel trends holds. Say so, and link the sensitivity tutorial
   (`05_honest_did`) and the pre-test power tutorial (`07_pretrends_power`).
6. **Control groups.** Never-treated and not-yet-treated comparisons rest on
   different identifying assumptions; neither is uniformly weaker. Do not
   describe one as "the stronger assumption".
7. **Event studies: one idiom.**

   ```python
   es = results.aggregate("event_study")   # if the results class has aggregate()
   es = results                            # otherwise (SunAbraham, a TWFE event_study=True fit)
   plot_event_study(es)
   ```

   Read the table with `es.to_dataframe()` when `es` is an `EventStudyResults`
   container, and with `results.to_dataframe(level="event_study")` otherwise
   (SunAbraham, WooldridgeDiD). Do not read raw `event_study_effects` dicts,
   rename the unified columns, or hand-draw a single estimator's event study in
   matplotlib; build multi-estimator overlays from the unified frames. Event
   time 0 is the first treated period - an estimator with a different native
   index states the mapping. The idiom is pinned on every event-study producer
   by `tests/test_visualization.py::TestEventStudyIdiomContract`.
8. **Warnings.** No unqualified `warnings.filterwarnings("ignore")` or
   `simplefilter("ignore")`. A warning that appears in committed output is
   either removed by fixing the call or explained in the next markdown cell.
   Two narrow exceptions: a scoped `warnings.catch_warnings()` with `category=`
   and `message=` for documented platform noise, and a category-qualified
   filter with a comment for a third-party framework's deprecation noise.
9. **Closing.** "Summary" (three to five bullets), "When to use it and when not
   to", "Next steps" linking other tutorials, and references. Business-track
   notebooks also keep their stakeholder-summary section.

### Mechanics

10. **Code style.** Import from top-level `diff_diff` (a submodule import only
    for a symbol that is not exported there). Use full class names, not the
    short aliases. Import matplotlib plainly - no `HAS_MATPLOTLIB` guard. Print
    results with `print(results.summary())` and read them as `results.att`,
    `.se`, `.conf_int`, `.p_value`.
11. **Reader voice.** No version or migration narration, deprecation-ledger
    ids, PR numbers, internal phase or wave names, internal paths, or
    `file:line` references - the migration guide and the changelog own that
    history. Hyphens, not em dashes, in cell sources. Sentence-case,
    unnumbered section headings.
12. **Committed outputs.** Every notebook is committed executed: all code
    cells run top to bottom in one pass (execution counts `1..n`), no error
    outputs, no stray empty figures. `nbsphinx_execute = "never"`, so the
    committed outputs are what Read the Docs shows.
13. **Drift test.** Every notebook has a `tests/test_t<NN>_<slug>_drift.py`
    that re-derives the numbers its prose quotes and asserts they appear in the
    rendered notebook (helpers in `tests/_tutorial_drift.py`). Prose that would
    need an unpinned number should point at the output instead.
14. **Numbering.** The number in a filename is a stable id, not a reading
    order (two notebooks are numbered 16). Reading order is the group order in
    `docs/tutorials/index.rst`.

Three notebooks keep documented exceptions: `16_survey_did` reads NHANES from
the tracked benchmark CSV (there is no loader for it), with a CDC download
only as the fallback; `29_mmm_calibration_pymc` and
`30_mmm_calibration_meridian` have frozen code cells, because their committed
outputs come from long MCMC runs in framework-specific environments.

### Re-executing a notebook

Committed outputs come from the Rust backend. Kernels are bound to a virtual
environment, so register one for the checkout you are working in:

```bash
python -m ipykernel install --user --name my-diffdiff-kernel
```

Execute into a scratch directory, compare with what is committed, then copy
the executed file back (never execute in place, and never pipe the run through
`tail`):

```bash
DIFF_DIFF_BACKEND=rust python - <<'PY'
from pathlib import Path

import nbformat
from nbclient import NotebookClient

out = Path("/tmp/executed")
out.mkdir(parents=True, exist_ok=True)   # before the (possibly long) run
nb = nbformat.read("docs/tutorials/01_basic_did.ipynb", as_version=4)
NotebookClient(nb, timeout=3000, kernel_name="my-diffdiff-kernel",
               resources={"metadata": {"path": "docs/tutorials"}}).execute()
nbformat.write(nb, out / "01_basic_did.ipynb")
PY
```

Read every changed output against the prose around it, then update the
notebook's drift test (and, for `31`-`33`, its code-cell hash pins). Before
opening the PR, mirror CI with
`DIFF_DIFF_BACKEND=python pytest --nbmake docs/tutorials/<notebook>.ipynb`.

## Test Writing Guidelines

### For Fallback/Error Handling Paths

- Don't just test that code runs without exception
- Assert the expected behavior actually occurred
- Bad: `result = func(bad_input)` (only tests no crash)
- Good: `result = func(bad_input); assert np.isnan(result.coef)` (tests behavior)

### For New Parameters

- Test parameter appears in `get_params()` output
- Test `set_params()` modifies the attribute
- Test parameter actually affects behavior (not just stored)

### For Warnings

- Capture warnings with `warnings.catch_warnings(record=True)`
- Assert warning message was emitted
- Assert the warned-about behavior occurred

### For NaN Inference Tests

Use `assert_nan_inference()` from conftest.py to validate ALL inference fields are
NaN-consistent. Don't check individual fields separately.

## Implementation Guidelines

### Adding an estimator parameter

Storing a parameter is the easy half. Before implementing, grep for every place it
will have to be honoured:

```bash
grep -rn "self\.<param>" diff_diff/<module>.py
```

A new parameter is only complete when it is:

- stored on `self` under its own name — `get_params()`/`set_params()` then
  cover it automatically (the shared `BaseEstimator` mixin introspects the
  `__init__` signature; a param stored under a DIFFERENT attribute name needs
  a `_PARAM_ATTR_ALIASES` entry, and constructor-derived state a
  `_DERIVED_CONFIG_ATTRS` entry — see `diff_diff/_base.py` and the
  inheritance map in `CLAUDE.md`)
- applied in **every** parameter-applying aggregation mode — `simple`, `event_study`, and `group`. `total` accepts no independent aggregation parameters, but every parameter that changes the overall keeper set or aggregation mass (e.g. `anticipation`) must also be reflected — and tested — in `total`'s mass
- applied in the **bootstrap/inference** paths, not just the analytical one
- reflected on the result object, so `to_dict()`/`summary()` do not misreport it
- propagated to the estimators that inherit it: `TwoWayFixedEffects` defines
  no `__init__` of its own, so it inherits a new `DifferenceInDifferences`
  constructor parameter automatically (the `MultiPeriodDiD` 3.9 shim that
  mirrored the signature was removed at the 4.0 cut);
  `SyntheticDiD` defines its OWN signature (it forwards only `alpha` to
  `super().__init__`, pinning `cluster=None`), so a new parent
  parameter must be explicitly mirrored — or explicitly rejected, like its
  `vcov_type`/`conley_*` guards — there, and `BaseEstimator` introspects the
  concrete signature, so an unmirrored parent param simply won't appear in
  `SyntheticDiD.get_params()`

The recurring bug is a parameter that works in the default aggregation and is
silently ignored in one of the others. The same trace applies when adding a new
*mode* or code path: follow it through aggregation, bootstrap, and results before
declaring it done.

### Control/comparison-group composition on new code paths

When you add a new mode or code path (e.g. `base_period="varying"`, a new
control-selection rule), **verify the control/comparison group is composed correctly
for that path**, not just the default. In staggered designs especially, confirm the
"not-yet-treated" comparison group excludes the treatment cohort itself, and that
parameter interactions (new mode × each aggregation method) select the intended
group. A silently mis-composed comparison group produces a plausible but wrong effect
with no error.

### Protecting arithmetic

Wrap **all** related operations in `np.errstate()`, not just the one that raised.
A guard around the final division still lets an upstream matrix multiply or
subtraction emit the warning. Include division, matrix multiplication, and any
operation that can overflow or underflow.

### Fixing a pattern that appears in more than one place

Grep for every occurrence **before** fixing any of them, and fix them all in the
same PR. Incremental fixes across review rounds are how a pattern bug survives:
each round looks resolved, and the next reviewer finds the sibling you missed.
If the same class of finding comes back a third time, stop patching sites and
name the invariant instead.
