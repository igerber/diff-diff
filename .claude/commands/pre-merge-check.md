---
description: Run pre-merge checks before submitting a PR
argument-hint: ""
---

# Pre-Merge Check

Run automated checks and display the pre-merge checklist before submitting a PR.

## Instructions

### 1. Identify Changed Files

Get the list of files that will be included in the PR:

```bash
# Get all changed files (tracked modifications + staged + untracked)
git diff --name-only HEAD
git diff --cached --name-only
git ls-files --others --exclude-standard
```

Categorize files into:
- **Methodology files**: `diff_diff/**/*.py` (excluding `__init__.py`) — match
  **recursively**. `diff_diff/` contains real packages (`visualization/`, `guides/`),
  and a non-recursive `diff_diff/*.py` glob silently skips every file inside them,
  so their pattern checks and test lookups never run.
- **Test files**: `tests/*.py`
- **Documentation files**: `*.md`, `*.rst`, `docs/**`
- **Notebooks**: `docs/tutorials/*.ipynb`

For a nested file, the name used for test discovery in Section 2.2 is the
**package directory**, not the module: `diff_diff/visualization/_event_study.py`
resolves on `visualization`.

### 2. Run Automated Pattern Checks (via the argv-safe helper)

#### 2.1 Methodology pattern checks + test resolution — `premerge_scan.py`

> **Canonical** — This section is referenced by `/submit-pr` and `/push-pr-update`.
> The single source of truth for the methodology pattern checks is
> `.claude/scripts/premerge_scan.py`. Do NOT hand-run greps over changed filenames.

**Why a helper, not prose greps.** A filename is git-controlled data: git permits
`$()`, backticks, quotes, spaces, and newlines in a path, so `grep pattern <files>` or
`pytest <files>` over a changed path executes a payload like `diff_diff/$(touch x).py`.
Prose "screens" leak (untracked paths, argument-vs-assignment forms, other commands).
`premerge_scan.py` closes this structurally: it discovers changed files via
`git … -z` through a subprocess **argv array** (never a shell), runs the pattern checks
(A–D below) as **pure-Python regex over file content** (opening a file by path cannot
execute a filename), resolves test coverage by matching names in Python, and **screens
every path** — emitting only validated-safe run-lists. It is unit-tested
(`tests/test_premerge_scan.py`) with staged *and* untracked `$(touch sentinel)`
filenames asserting nothing executes.

Run it:

```bash
SCRATCH="$(git rev-parse --git-path premerge-scan)"; mkdir -p "$SCRATCH"
python3 .claude/scripts/premerge_scan.py --scratch "$SCRATCH"
```

- If it **exits 4**, a git or file-read operation failed — the scan is **incomplete**
  and its run-lists were truncated to empty. **Stop and report the error;** do NOT
  continue (empty run-lists would silently run no tests and misstate coverage).
- If it **exits 3**, it found a changed path containing shell metacharacters. It has
  already excluded that path; surface the reported path(s) for manual review — a
  methodology filename with `$()`/backticks is itself a red flag — and do not improvise
  a shell command over it.
- Report its pattern findings (file:line) to the user.
- The safe, NUL-delimited test/notebook run-lists it writes
  (`$SCRATCH/run-tests.z`, `$SCRATCH/run-notebooks.z`) are consumed argv-safely in
  Section 4 — never re-derive filenames into a shell command yourself.

The checks the helper performs (kept here as the human-readable spec; the helper is the
executable source of truth):

- **Check A** — inline inference: a line matching `t_stat = … / se` without
  `safe_inference`. Fix: use `safe_inference()` from `diff_diff.utils`.
- **Check B** — zero-SE fallback: `if se > 0 … else 0.0`. Fix: SE=0 → NaN, not 0.0.
- **Check C** — a new `self.X` (from the diff) absent from `get_params()`.
- **Check D** — `compute_confidence_interval` without a `safe_inference`/`isfinite`/
  `if se >` guard.
- **Test resolution** — for each changed methodology file, the test files whose name
  contains its stem (leading underscore stripped) or its `doc-deps.yaml` group name;
  reports any with no resolved suite so coverage is confirmed by hand rather than
  silently skipped.

#### 2.3 Docstring Check (heuristic)

Public functions in the changed `.py` files should have docstrings, and functions with
changed signatures should have up-to-date `Parameters` sections. This is a heuristic —
do **not** grep changed filenames in a shell (a path like `diff_diff/$(touch x).py`
would execute). Read each changed methodology file (the safe list from Section 2.1)
with the Read tool and scan for public `def`s lacking a docstring, and for `+`-added
`def` lines in the diff whose docstring may be stale. Flag those for the user to
confirm. Reading a file by path never executes its name.

#### 2.5 Documentation Impact Check

If any source files in `diff_diff/` changed, read `docs/doc-deps.yaml` and identify which
dependent documentation files are NOT also in the changed file set (from Section 1).

For each changed source file:
1. Look up its entry in `docs/doc-deps.yaml` (resolving group membership for multi-file modules)
2. Check each dependent doc's `path` against the changed file set
3. Report docs that were NOT changed as warnings:
   - ALL docs with `type: methodology` (regardless of `drift_risk`) — methodology deviations
     are P1 in AI review, so this warning must always fire
   - All HIGH `drift_risk` docs (any type)

**Report format**:
```
Documentation impact: source files changed but related docs were not updated:
  [METHODOLOGY] docs/methodology/REGISTRY.md -- <section hint>
  [HIGH] docs/survey-roadmap.md
  [MEDIUM] README.md -- <section hint> (N more -- run /docs-impact for details)
```

This is a WARNING, not a blocker — not every source change requires a doc update.
For full details, run `/docs-impact`.

#### 2.6 Secret Scanning Patterns (Canonical Definitions)

> These patterns are referenced by `/submit-pr` and `/push-pr-update`.

**Content pattern** (use with `-G` flag, `--name-only` to avoid leaking secrets):
```bash
-G "(AKIA[A-Z0-9]{16}|ghp_[a-zA-Z0-9]{36}|sk-[a-zA-Z0-9]{48}|gho_[a-zA-Z0-9]{36}|[Aa][Pp][Ii][_-]?[Kk][Ee][Yy][[:space:]]*[=:]|[Ss][Ee][Cc][Rr][Ee][Tt][_-]?[Kk][Ee][Yy][[:space:]]*[=:]|[Pp][Aa][Ss][Ss][Ww][Oo][Rr][Dd][[:space:]]*[=:]|[Pp][Rr][Ii][Vv][Aa][Tt][Ee][_-]?[Kk][Ee][Yy]|[Bb][Ee][Aa][Rr][Ee][Rr][[:space:]]+[a-zA-Z0-9_-]+|[Tt][Oo][Kk][Ee][Nn][[:space:]]*[=:])"
```

**Sensitive filename pattern**:
```bash
grep -iE "(\.env|credentials|secret|\.pem|\.key|\.p12|\.pfx|id_rsa|id_ed25519)$"
```

**Usage**: Apply content pattern to `--cached` for staged changes, or `<ref>..HEAD` for already-committed changes. Always use `--name-only` and `|| true`.

### 3. Display Context-Specific Checklist

Based on what changed, display the appropriate checklist items:

#### Always Show (Core Checklist)
```
## Pre-Merge Checklist

Based on your changes to: <list of changed files>

### Behavioral Completeness
- [ ] Happy path tested
- [ ] Edge cases tested (empty data, NaN inputs, boundary conditions)
- [ ] Error/warning paths tested with behavioral assertions
```

#### If Methodology Files Changed
```
### Inference Field Consistency
- [ ] If SE can be 0/undefined, ALL inference fields (t-stat, p-value, CI) return NaN
- [ ] Aggregation methods propagate NaN correctly
- [ ] Bootstrap methods handle NaN in base estimates

### Control Group Logic (if adding new modes/code paths)
- [ ] Control group composition verified for new code paths
- [ ] "Not-yet-treated" excludes the treatment cohort itself
- [ ] Parameter interactions tested with all aggregation methods

### Methodology Deviation Documentation
- [ ] If deviating from reference implementation: added a reviewer-recognized label
      (`**Note:**`, `**Deviation from R:**`, or `**Note (deviation from R):**`) in REGISTRY.md
- [ ] No undocumented methodology deviations (AI reviewer flags these as P1)
```

#### If Documentation Files Changed
```
### Documentation Sync
- [ ] Docstrings updated for changed function signatures
- [ ] `diff_diff/guides/llms.txt` updated if the public API surface changed (AI-agent contract)
- [ ] `docs/api/*.rst` and `docs/references.rst` updated as appropriate
- [ ] `README.md` updated ONLY for landing-page-relevant changes (catalog one-liner, hero/badges/tagline, top-level capability paragraph). Per CONTRIBUTING.md, README is not the place for usage examples or per-estimator sections.
```

#### If This Appears to Be a Bug Fix
```
### Pattern Consistency (Bug Fix)
- [ ] Grepped for similar patterns across codebase before fixing
- [ ] Fixed ALL occurrences, not just the one that was reported
- [ ] Verified fix doesn't break other code paths
```

### 4. Ask About Running Tests

Use AskUserQuestion. **Targeted is the default** — run tests for the touched code
areas, not the whole suite:

```
Would you like to run tests now?

Options:
1. Yes - tests for changed files only (recommended)
2. No - skip tests for now
3. Yes - full test suite (slow; only when the change is broad)
```

For option 1, run the resolved suites `premerge_scan.py` wrote to
`$SCRATCH/run-tests.z` (Section 2.1) — **argv-safe via `xargs -0`**, so a hostile test
filename is passed as an argument, never reparsed by the shell:

```bash
SCRATCH="$(git rev-parse --git-path premerge-scan)"
if [ -s "$SCRATCH/run-tests.z" ]; then
  xargs -0 pytest < "$SCRATCH/run-tests.z"
else
  echo "No test files resolved for the changed modules."
  # Offer: run the full suite (explicitly), name the tests to run, or skip.
  # Do NOT run a bare `pytest` — it discovers and runs the ENTIRE suite, the opposite
  # of the targeted run requested.
fi
```

The list already includes every changed `tests/test_*.py` and the module-resolved
suites (helper's job). Never run more than one pytest process at a time.

### 4b. Validate Notebooks (only if notebooks changed)

Skip this section entirely when no `docs/tutorials/*.ipynb` files changed.

CI runs `pytest --nbmake docs/tutorials/` but is gated behind the `ready-for-ci`
label, and the branch is frozen once that label is on — so notebook breakage has to
be caught *here*, before submitting. Two notebooks are additionally excluded from
the main CI job as too slow and are only ever executed locally:
`06_power_analysis.ipynb` and `10_trop.ipynb`.

Run the changed notebooks from the helper's safe list — **argv-safe via `xargs -0`**,
never a filename pasted into the command:

```bash
SCRATCH="$(git rev-parse --git-path premerge-scan)"
[ -s "$SCRATCH/run-notebooks.z" ] && \
  xargs -0 pytest --nbmake --nbmake-timeout=600 < "$SCRATCH/run-notebooks.z"
```

`run-notebooks.z` already contains **every** changed notebook, `06_power_analysis.ipynb`
and `10_trop.ipynb` included — so this one command covers them; do **not** run those two
again separately. The point about 06/10 is only *why* local validation matters (CI skips
them), not that they need a second invocation.

**Exception — `26_composition_drift_calibration.ipynb` needs a dependency the dev
extra does not provide.** It imports `balance`, which is not in `[dev]`; CI runs it
in a separate interop job that installs `balance>=0.21`. Running it in a normal dev
environment fails on the import, which is an environment problem masquerading as a
notebook regression.

If that notebook is among the changed set, check first and ask before installing:

```bash
python -c "import balance" 2>/dev/null && echo "balance: present" || echo "balance: MISSING"
```

If missing, offer to `pip install "balance>=0.21"` or to skip that one notebook and
let the CI interop job cover it. Do not install it silently — it is a heavy
dependency that is deliberately absent from `[dev]`.

`nbmake` executes notebooks **without writing outputs back**, so there is no
clear-outputs cleanup step and nothing dirty can reach a commit. Use it rather than
`jupyter nbconvert --execute --inplace`, which mutates files in place and is not a
dependency of this project.

Notebook runs take minutes. Ask before starting, and report per-notebook pass/fail.

### 5. Report Summary

```
## Pre-Merge Check Complete

### Automated Checks
- Pattern checks: [PASS/WARN - N potential issues found]
- Test coverage: [PASS/WARN - N methodology files without test changes]

### Manual Checklist
Review the checklist items above before running /submit-pr

### Findings to Address
<list any warnings from pattern checks>

### Next Steps
- Address any warnings above
- Complete manual checklist items
- When ready: /submit-pr "Your PR title"
```

## Notes

- Non-mutating: analyses and reports only. Running tests and `--nbmake` does not
  modify tracked files.
- Run this BEFORE `/submit-pr` to catch issues early
- Pattern checks are heuristics - review flagged items manually to confirm
- If pattern checks find issues, fix them before submitting
- Check B (zero-SE fallback) currently matches nothing repo-wide. That is the point:
  it is a regression guard against reintroducing an anti-pattern that was eradicated,
  not a dead check. Keep it.
