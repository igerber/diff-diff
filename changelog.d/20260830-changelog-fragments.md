### Internal
- **Changelog fragments**: release notes are now authored as per-PR files
  under `changelog.d/` (see `changelog.d/README.md`) instead of editing
  `CHANGELOG.md`'s `## [Unreleased]` section, which stays pointer-only
  between releases (CI-enforced by `tests/test_changelog_fragments.py`) so
  concurrent PRs no longer conflict on the changelog. At release,
  `.claude/scripts/changelog_compile.py compile` merges the fragments into
  the new version section and deletes them. One ordering change relative to
  the old convention: within a category, compiled release sections list
  entries oldest-first (ascending fragment-filename order) rather than the
  newest-first order that prepending into Unreleased produced.
