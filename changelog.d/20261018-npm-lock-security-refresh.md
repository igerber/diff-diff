### Security
- **CI smoke-test lockfile refresh**: bumped the transitive `undici` (7.29.0 -> 7.30.1)
  and `source-map-js` (1.2.1 -> 1.2.2) pins in `.github/scripts/package-lock.json`,
  clearing all seven open Dependabot alerts. CI-only (docs search-excerpt smoke test);
  nothing in the published wheel is affected.

### Internal
- **Dependabot npm coverage**: added an `npm` ecosystem entry for `/.github/scripts`
  so the smoke-test toolchain gets weekly version-update PRs (grouped minor/patch)
  instead of drifting into advisories between manual lockfile refreshes.
