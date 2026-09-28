# Changelog fragments

Release notes are authored here as **one file per PR** instead of editing
`CHANGELOG.md`'s `## [Unreleased]` section (which stays pointer-only between
releases — CI-enforced by `tests/test_changelog_fragments.py`, so concurrent
PRs never conflict on the changelog). At release time
`.claude/scripts/changelog_compile.py compile` merges every fragment into the
new `## [X.Y.Z] - DATE` section and deletes the fragment files.

## Filename

`YYYYMMDD-<slug>.md` — e.g. `20260830-dml-s42-fixtures.md`.

- `YYYYMMDD` = authoring date (must be a real calendar date).
- `<slug>` = kebab-case topic, usually derived from the branch name; keep it
  unique per PR. (Two same-day PRs picking the identical slug produce an
  add/add conflict on a brand-new file — trivially resolved by renaming one,
  and strictly cheaper than the same-line CHANGELOG conflicts this system
  replaces.)
- Within a category, fragments compile **oldest-first** (ascending filename
  sort).

## Body

One or more category blocks, written exactly as the bullets should appear in
`CHANGELOG.md` (same voice and style as existing entries):

```markdown
### Fixed
- **Headline of the fix**: details, wrapped with
  two-space continuation lines.
  - nested sub-bullets are fine too.
```

Rules (validated by `changelog_compile.py check`):

- A block starts with `### <Category>` and must contain at least one
  top-level bullet (`- ` at column 0). Wrapped prose and nested sub-bullets
  are indented continuation lines. Blank lines are allowed anywhere.
- Nothing outside category blocks; no `## ` headers (at any CommonMark
  indentation or spacing).
- No fenced code blocks or raw-HTML blocks (` ``` `, `~~~`, `<pre>`, …) and
  no version-link-like constructs (`[X.Y.Z]:`): the compiled changelog
  refuses both (they can swallow or outrank the release headers and the
  comparison-link block), and `check` rejects them at PR time so a fragment
  never blocks the release. Use inline code instead of fences.
- Allowed categories (this list mirrors the compiler constant in
  `.claude/scripts/changelog_compile.py`; the mirror is test-pinned):
  `Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`, `Security`,
  `Performance`, `Documentation`, `Testing`, `Breaking Changes`,
  `Behavioral Changes`, `Internal`.
- 4.0-program PRs must still name the flipped `M-xxx` row ids in the
  fragment (the `docs/v4-design.md` per-PR obligation).

## Rebasing an older branch

A branch created before this system that added an `## [Unreleased]` bullet
will conflict on rebase and then fail the pointer-only guard: move your
bullet into a new fragment file here and leave the Unreleased section as the
pointer comment.
