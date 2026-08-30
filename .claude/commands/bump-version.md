---
description: Update version numbers across codebase and ensure CHANGELOG is populated
argument-hint: "<version> (e.g., 2.2.0)"
---

# Bump Version

Update version numbers across the codebase and ensure CHANGELOG is properly populated for a new release.

## Arguments

The user must provide a version number: `$ARGUMENTS`

- If empty or not provided: Ask the user for the target version
- Otherwise: Use the provided version (must match semver pattern X.Y.Z)

## Version Locations

Files that need updating:

| File | Format | Line |
|------|--------|------|
| `diff_diff/__init__.py` | `__version__ = "X.Y.Z"` | ~134 |
| `pyproject.toml` | `version = "X.Y.Z"` | ~7 |
| `rust/Cargo.toml` | `version = "X.Y.Z"` | ~3 |
| `CHANGELOG.md` | Section header + comparison link | Top + bottom |
| `diff_diff/guides/llms-full.txt` | `- Version: X.Y.Z` | ~5 |
| `CITATION.cff` | `version: "X.Y.Z"` + `date-released: "YYYY-MM-DD"` | ~10, ~11 |

## Instructions

1. **Parse and validate version**:
   - If no argument provided, use AskUserQuestion to get the target version
   - Validate format matches semver pattern `X.Y.Z` (e.g., `2.2.0`, `3.0.0`, `1.10.5`)
   - If invalid, ask user to provide a valid version

2. **Get current version**:
   - Read `diff_diff/__init__.py` and extract the current `__version__` value
   - Store as `OLD_VERSION` for comparison link generation

3. **Compile the changelog and resolve `RELEASE_DATE`** (release notes come
   exclusively from `changelog.d/` fragments; the old git-log generation step
   is removed):

   ```bash
   python3 .claude/scripts/changelog_compile.py compile --version NEW_VERSION --date "$(date +%F)"
   ```

   Key off the exit code:
   - **Exit 0** (compiled): the `## [NEW_VERSION]` section and comparison link
     were written and the fragments deleted. `RELEASE_DATE` = today (the
     `--date` you passed).
   - **Exit 4** (already compiled): the section already exists with content and
     no fragments remain — an idempotent re-run after a partial bump. The
     compiler prints the existing header's date; use THAT as `RELEASE_DATE`.
     On such a re-run, verify EACH version file in the table individually
     (step 5's blind `OLD_VERSION → NEW_VERSION` replacement no-ops once
     `diff_diff/__init__.py` is already bumped, so grep every file rather than
     trusting the replacements).
   - **Any other exit**: surface the compiler's message and stop. For a
     legitimately fragment-free cycle (all merged PRs CI/tooling-only), write
     a minimal `### Internal` stub fragment describing the release, **commit
     it**, then re-run (an uncommitted fragment is rejected by the compiler's
     dirty-fragment guard — deliberately, so nothing unreviewed is swept into
     a release and deleted).

   `RELEASE_DATE` is the single source of truth for the release date across every file
   touched in this bump. Do not recompute it downstream.

4. *(Removed — the compiler owns changelog generation; the fragment format is
   documented in `changelog.d/README.md`.)*

5. **Update version in all files**:
   Use the Edit tool to update each file:

   - `diff_diff/__init__.py`:
     Replace `__version__ = "OLD_VERSION"` with `__version__ = "NEW_VERSION"`

   - `pyproject.toml`:
     Replace `version = "OLD_VERSION"` with `version = "NEW_VERSION"`

   - `rust/Cargo.toml`:
     Replace `version = "OLD_VERSION"` (the first version line under [package]) with `version = "NEW_VERSION"`
     Note: Rust version may differ from Python version; always sync to the new version

   - `diff_diff/guides/llms-full.txt`:
     Replace `- Version: OLD_VERSION` with `- Version: NEW_VERSION`

   - `CITATION.cff`:
     Replace `version: "OLD_VERSION"` with `version: "NEW_VERSION"`.
     Also update `date-released: "OLD_DATE"` to `date-released: "{RELEASE_DATE}"`
     using the `RELEASE_DATE` resolved in step 3. Both fields are quoted strings;
     preserve the quoting style. `RELEASE_DATE` must match the CHANGELOG header
     date; never substitute a freshly computed "today" value here.

6. **CHANGELOG comparison link** — written by the compiler in step 3 (format
   `[NEW]: <base>/compare/vOLD...vNEW`, both versions `v`-prefixed, inserted
   immediately above the previous version's link line; the base URL is taken
   from that line, not from the git remote). Nothing to do manually.

7. **Report summary**:
   Display a summary of all changes made:
   ```
   Version bump complete: OLD_VERSION -> NEW_VERSION

   Files updated:
   - diff_diff/__init__.py: __version__ = "NEW_VERSION"
   - pyproject.toml: version = "NEW_VERSION"
   - rust/Cargo.toml: version = "NEW_VERSION"
   - diff_diff/guides/llms-full.txt: Version: NEW_VERSION
   - CITATION.cff: version: NEW_VERSION, date-released: YYYY-MM-DD
   - CHANGELOG.md: compiled [NEW_VERSION] from changelog.d/

   Next steps:
   1. Review changes: git diff
   2. Commit: git commit -am "Bump version to NEW_VERSION"
   3. Tag: git tag vNEW_VERSION
   4. Push: git push && git push --tags
   ```

## Notes

- The Rust version in `rust/Cargo.toml` is always synced to match the Python version
- If CHANGELOG already has the target version section (and `changelog.d/` is empty),
  the compiler exits 4 and the bump proceeds as a re-run — the existing section is
  never overwritten
- Release notes come from curated `changelog.d/` fragments; commit messages are never read
- `CITATION.cff` `date-released` and the `CHANGELOG.md` section header share a single
  `RELEASE_DATE` resolved in step 3: an already-compiled header's date wins via the
  compiler's exit-4 path (so a re-run after a partial bump doesn't silently drift
  from the CITATION date); otherwise today's date is used for both. If the release
  is cut on a different day than the bump, update both surfaces manually — drift
  causes auto-citation tools (Zenodo, GitHub's "cite this repository", reference
  managers) to report stale metadata.
