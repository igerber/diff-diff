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
   - New notebooks are registered in `docs/tutorials/index.rst` (NOT `docs/index.rst`):
     add a toctree entry with a short display label to the matching group
     (Business Applications / Fundamentals / Advanced Methods / Study Design)
     plus a `grid-item-card` in that group's card grid

4. **`docs/references.rst`** (bibliography source of truth) - Add:
   - Full citation under the appropriate sub-section (matches the `### Subsection` headings already in that file)
   - Use the RST format: `**Author (Year).** "Title." *Journal*, vol(num), pages. <https://doi.org/X>`

5. **`README.md`** - Add ONLY:
   - One line in the `## Estimators` catalog with the paper citation and RTD link

6. **`CHANGELOG.md`** - Add a release-note bullet under the next unreleased version.

7. **`CLAUDE.md`** - Update only if adding new critical rules or design patterns.

8. **`ROADMAP.md`** - Update only if shipping moves an item from planned to current.

9. **`docs/doc-deps.yaml`** - Add source-to-doc mappings for the new module.

### For Bug Fixes or Minor Enhancements

- Update relevant docstrings
- Add/update tests
- Update `CHANGELOG.md`
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
