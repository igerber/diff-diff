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

## Instructions

1. **Parse and validate version**:
   - If no argument provided, use AskUserQuestion to get the target version
   - Validate format matches semver pattern `X.Y.Z` (e.g., `2.2.0`, `3.0.0`, `1.10.5`)
   - If invalid, ask user to provide a valid version

2. **Get current version**:
   - Read `diff_diff/__init__.py` and extract the current `__version__` value
   - Store as `OLD_VERSION` for comparison link generation

3. **Check CHANGELOG entry**:
   - Search `CHANGELOG.md` for `## [NEW_VERSION]` section header
   - If found: Verify it has content (at least one `### Added/Changed/Fixed` subsection with bullet points)
   - If not found or empty: Generate entry from git commits (step 4)
   - If found with content: Skip to step 5

4. **Generate CHANGELOG from git** (only if needed):
   - Run: `git log v{OLD_VERSION}..HEAD --oneline`
   - If no tag exists, use: `git log --oneline -50`
   - Categorize commits using these heuristics:
     - **Added**: commits containing "add", "new", "implement", "introduce", "create"
     - **Changed**: commits containing "update", "change", "improve", "optimize", "refactor", "enhance"
     - **Fixed**: commits containing "fix", "bug", "correct", "repair", "resolve"
   - Get today's date in YYYY-MM-DD format
   - Create CHANGELOG entry in this format:
     ```markdown
     ## [X.Y.Z] - YYYY-MM-DD

     ### Added
     - Feature description from commit message

     ### Changed
     - Change description from commit message

     ### Fixed
     - Fix description from commit message
     ```
   - Only include sections that have commits (omit empty sections)
   - Insert the new entry after the changelog header (after the "adheres to Semantic Versioning" line)

5. **Update version in all files**:
   Use the Edit tool to update each file:

   - `diff_diff/__init__.py`:
     Replace `__version__ = "OLD_VERSION"` with `__version__ = "NEW_VERSION"`

   - `pyproject.toml`:
     Replace `version = "OLD_VERSION"` with `version = "NEW_VERSION"`

   - `rust/Cargo.toml`:
     Replace `version = "OLD_VERSION"` (the first version line under [package]) with `version = "NEW_VERSION"`
     Note: Rust version may differ from Python version; always sync to the new version

6. **Update CHANGELOG comparison links**:
   - At the bottom of `CHANGELOG.md`, after `[OLD_VERSION]:`, add the new comparison link:
     ```
     [NEW_VERSION]: https://github.com/igerber/diff-diff/compare/vOLD_VERSION...vNEW_VERSION
     ```

7. **Report summary**:
   Display a summary of all changes made:
   ```
   Version bump complete: OLD_VERSION -> NEW_VERSION

   Files updated:
   - diff_diff/__init__.py: __version__ = "NEW_VERSION"
   - pyproject.toml: version = "NEW_VERSION"
   - rust/Cargo.toml: version = "NEW_VERSION"
   - CHANGELOG.md: Added/verified [NEW_VERSION] entry

   Next steps:
   1. Review changes: git diff
   2. Commit: git commit -am "Bump version to NEW_VERSION"
   3. Tag: git tag vNEW_VERSION
   4. Push: git push && git push --tags
   ```

## Notes

- The Rust version in `rust/Cargo.toml` is always synced to match the Python version
- If CHANGELOG already has the target version entry with content, it will not be overwritten
- Commit messages are cleaned up (prefixes like "feat:", "fix:" are removed) for CHANGELOG
- The comparison link format uses `v` prefix for tags (e.g., `v2.2.0`)
