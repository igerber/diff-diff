# Contributing to diff-diff

## Documentation Requirements

When implementing new functionality, **always include accompanying documentation updates**.

### For New Estimators or Major Features

1. **README.md** - Add:
   - Feature mention in the features list
   - Full usage section with code examples
   - Parameter documentation table
   - API reference section (constructor params, fit() params, results attributes/methods)
   - Scholarly references if applicable

2. **docs/api/*.rst** - Add:
   - RST documentation with `autoclass` directives
   - Method summaries
   - References to academic papers

3. **docs/tutorials/*.ipynb** - Update relevant tutorial or create new one:
   - Working code examples
   - Explanation of when/why to use the feature
   - Comparison with related functionality

4. **CLAUDE.md** - Update only if adding new critical rules or design patterns

5. **ROADMAP.md** - Update:
   - Move implemented features from planned to current status
   - Update version numbers

### For Bug Fixes or Minor Enhancements

- Update relevant docstrings
- Add/update tests
- Update CHANGELOG.md (if exists)
- **If methodology-related**: Update `docs/methodology/REGISTRY.md` edge cases section

### Scholarly References

For methods based on academic papers, always include:
- Full citation in README.md references section
- Reference in RST docs with paper details
- Citation in tutorial summary

Example format:
```
Sun, L., & Abraham, S. (2021). Estimating dynamic treatment effects in
event studies with heterogeneous treatment effects. *Journal of Econometrics*,
225(2), 175-199.
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
