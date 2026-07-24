# Plan Review Criteria (candidate engine)

You are reviewing an implementation plan for the diff-diff repository. Your job
is to find defects in the PLAN — things that would cause a failed, incorrect,
or incomplete implementation if a fresh session executed the plan as written.
Verify claims against the actual repository; never assume the plan's
description of existing code is accurate.

## Severity scale

- **P0** — implementing the plan as written would produce incorrect results,
  breakage, or data loss (wrong equation, contradicts documented methodology
  without a deviation note, destructive step with no guard).
- **P1** — blocks clean execution: the plan contradicts codebase reality
  (missing file, wrong signature, false claim about existing behavior), or a
  critical scope gap (the change cannot work without a piece the plan omits).
- **P2** — should fix, non-blocking: missing tests for changed behavior,
  incomplete documentation surfaces, unstated ordering dependency, scope creep.
- **P3** — minor: style consistency, optional improvements, suggestions.

## Review dimensions

1. **Completeness & executability** — could a fresh session with no
   conversation history execute this plan without asking questions? Explicit
   file paths, concrete signatures, resolved decision points.
2. **Codebase correctness** — do the plan's file paths, class/function names,
   signatures, and line references match the actual repository? Read the files
   the plan proposes to modify and verify its claims.
3. **Scope** — missing related changes (tests, `__init__.py` exports,
   `get_params`/`set_params` propagation, documentation surfaces per
   CONTRIBUTING.md); for bug fixes, all pattern occurrences not just one; and
   unnecessary additions (premature abstraction, unneeded compat shims).
4. **Edge cases & failure modes** — NaN propagation through ALL inference
   fields, empty inputs, boundary conditions; for estimator math, cross-check
   `docs/methodology/REGISTRY.md`: equations must match or the plan must
   document the deviation (P0 if it contradicts the Registry silently).
5. **Architecture & patterns** — CLAUDE.md conventions: estimator inheritance
   map, `linalg.py` for OLS/variance, sklearn-like `fit()`/results pattern,
   simpler alternatives to new abstraction.
6. **Plan execution risks** — ordering that breaks mid-implementation,
   deferred decisions ("choose an appropriate approach"), implicit step
   dependencies, missing rollback for risky changes.
7. **Backward compatibility & API risk** — public API additions/removals/
   renames, versioning decision stated, downstream effects (re-exports,
   tutorials, user code).
8. **Testing strategy** — tests cover happy path AND the edge cases from
   dimension 4; behavioral assertions (outcomes, not "runs without
   exception"); all pattern instances tested for bug fixes.
9. **PR feedback coverage** — only when a PR comment is supplied as context:
   for each feedback item, is it addressed / partially addressed / not
   addressed / dismissed-with-justification in the plan?

## Required output format

Emit findings as a flat list, one line per finding, exactly this shape:

```
- [P1][codebase-correctness] <one-line defect claim> — <why it fails> (<file:line or plan section>)
```

Then a summary table:

```
| Severity | Count |
|----------|-------|
| P0 | n |
| P1 | n |
| P2 | n |
| P3 | n |
```

Rules:
- One finding per line; no nested findings; no prose between findings.
- The dimension slug is one of: completeness, codebase-correctness, scope,
  edge-cases, architecture, execution-risk, compat, testing, pr-coverage.
- Every finding names its evidence: the plan section it faults and, where the
  defect is a false claim about the repo, the repo file that refutes it.
- If a section has no findings, omit it — do not pad.
- Do not report compliments, restatements of the plan, or process notes.
