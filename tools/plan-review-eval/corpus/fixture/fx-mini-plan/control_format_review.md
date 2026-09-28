<!-- Committed rehearsal artifact (fabricated): a review of the fixture plan
in the CONTROL engine's native format. Its ONLY mention of the seeded
nonexistent-symbol defect sits under "## Questions for the Author" — the
dress rehearsal's format-parity extraction probe feeds this file through the
extraction stage and fails if that defect does not survive as a finding
(DECISION_RULE.md, "format-parity extraction probe"). Not a corpus case: the
loader reads only case.json + fixture.plan. -->

## Overall Assessment

The plan extends the shared inference utility to emit a user-visible warning
when inference is computed from a degenerate variance. The direction is
consistent with the no-silent-failures convention, but the plan declares no
tests for the new behavior, and a symbol it builds on could not be located.

---

## Critical Issues

None.

---

## Medium Issues

MEDIUM #1: The plan adds new user-visible behavior (a warning emitted on
degenerate variance) but states that no tests are needed. Project
conventions require behavioral assertions for new behavior — a test should
assert the warning fires (`pytest.warns`) on the degenerate input and does
NOT fire on a healthy path.

---

## Low Issues

None.

---

## Conventions Checklist

- [ ] Tests planned for new behavior (see MEDIUM #1)
- [x] No new dependencies introduced
- [x] No methodology/REGISTRY.md surface touched

---

## Questions for the Author

1. Step 2 says the warning is raised from `safe_inference_v2()` in
   `diff_diff/utils.py`, but I could not find any function by that name in
   the module — only `safe_inference()` exists. Which function does the plan
   intend to modify? If `safe_inference_v2` is expected to exist, where is
   it introduced? Implementation cannot begin until this is resolved.
